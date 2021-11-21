import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras import backend as K

def huber_loss(y_true, y_pred):
    threshold = 1.
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold

    small_error_loss = tf.square(error) / 2 # MSE
    big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold)) # MAE

    return tf.where(is_small_error, small_error_loss, big_error_loss)

def between_eyes_angle_loss(y_true, y_pred):
    x1_gt = y_true[:, :, :, 0:1]
    x2_gt = y_true[:, :, :, 1:2]
    y1_gt = y_true[:, :, :, 2:3]
    y2_gt = y_true[:, :, :, 3:4]
    x1_gt = K.reshape(x1_gt, len(x1_gt))
    x2_gt = K.reshape(x2_gt, len(x2_gt))
    y1_gt = K.reshape(y1_gt, len(y1_gt))
    y2_gt = K.reshape(y2_gt, len(y2_gt))

    gradient = tf.math.divide(tf.math.subtract(y2_gt, y1_gt), tf.math.subtract(x2_gt, x1_gt))
    angle_gt = tf.math.atan(gradient)

    x1_pred = y_pred[:, :, :, 0:1]
    x2_pred = y_pred[:, :, :, 1:2]
    y1_pred = y_pred[:, :, :, 2:3]
    y2_pred = y_pred[:, :, :, 3:4]
    x1_pred = K.reshape(x1_pred, len(x1_pred))
    x2_pred = K.reshape(x2_pred, len(x2_pred))
    y1_pred = K.reshape(y1_pred, len(y1_pred))
    y2_pred = K.reshape(y2_pred, len(y2_pred))

    # get pred angle
    gradient = tf.math.divide(tf.math.subtract(y2_pred, y1_pred), tf.math.subtract(x2_pred, x1_pred))
    angle_pred = tf.math.atan(gradient)
    angle_gap = tf.math.abs(tf.math.subtract(angle_gt, angle_pred))
    loss = tf.math.sin(angle_gap)
    loss = tf.math.divide(loss, 10.) # MSE unit balancing

    loss = K.reshape(loss, (len(y2_pred), 1))
    return loss
def between_eyes_dist_loss(y_true, y_pred):
    x1_gt = y_true[:, :, :, 0:1]
    x2_gt = y_true[:, :, :, 1:2]
    y1_gt = y_true[:, :, :, 2:3]
    y2_gt = y_true[:, :, :, 3:4]
    x1_gt = K.reshape(x1_gt, len(x1_gt))
    x2_gt = K.reshape(x2_gt, len(x2_gt))
    y1_gt = K.reshape(y1_gt, len(y1_gt))
    y2_gt = K.reshape(y2_gt, len(y2_gt))
    x1_pred = y_pred[:, :, :, 0:1]
    x2_pred = y_pred[:, :, :, 1:2]
    y1_pred = y_pred[:, :, :, 2:3]
    y2_pred = y_pred[:, :, :, 3:4]
    x1_pred = K.reshape(x1_pred, len(x1_pred))
    x2_pred = K.reshape(x2_pred, len(x2_pred))
    y1_pred = K.reshape(y1_pred, len(y1_pred))
    y2_pred = K.reshape(y2_pred, len(y2_pred))

    gt_dist = K.sqrt(tf.math.add(K.square(tf.math.subtract(x1_gt, x2_gt)), K.square(tf.math.subtract(y1_gt, y2_gt))))
    pred_dist = K.sqrt(tf.math.add(K.square(tf.math.subtract(x1_pred, x2_pred)), K.square(tf.math.subtract(y1_pred, y2_pred))))
    loss = tf.math.divide(K.square(tf.math.subtract(gt_dist, pred_dist)), 2.)

    return loss

def hybrid_pool_layer(pool_size=(2,2)):
    def apply(x):
        l =  tf.keras.layers.Add()([tf.keras.layers.MaxPooling2D(pool_size)(x), tf.keras.layers.AveragePooling2D(pool_size)(x)])
        return tf.divide(l, 2.)
    return apply

def _load_datasets():
    print('LOADING datasets')
    x_train = np.load('x_train2.npy') / 255
    y_train = np.load('y_train2.npy') / 96
    x_test = np.load('x_test2.npy') / 255
    y_test = np.load('y_test2.npy') / 96
    print('x_train, x_test shape:', x_train.shape, x_test.shape)
    print('y_train, y_test shape:', y_train.shape, y_test.shape)
    y_train = np.reshape(y_train, (-1, 1, 1, 4))
    y_test = np.reshape(y_test, (-1, 1, 1, 4))
    print('y_train, y_test shape:', y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test

# USE THIS
def _hy_pool_Model_huber_mse():
    inputs = tf.keras.layers.Input(shape=(96, 96, 1))
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = hybrid_pool_layer((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = hybrid_pool_layer((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = hybrid_pool_layer((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = hybrid_pool_layer((2, 2))(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = hybrid_pool_layer((2, 2))(x)

    x = tf.keras.layers.Conv2D(4, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)

    outputs = tf.keras.layers.Conv2D(4, kernel_size=(3, 3), strides=1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='hy_pool')
    model.compile(loss=[huber_loss,  tf.keras.losses.mean_squared_error],
                  # loss_weights=[0.3, 0.7],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=[huber_loss, 'mse'],
                  run_eagerly=True)
    print(model.summary())
    return model

def _hy_pool_Model_mse():
    inputs = tf.keras.layers.Input(shape=(96, 96, 1))
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = hybrid_pool_layer((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = hybrid_pool_layer((2, 2))(x)

    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = hybrid_pool_layer((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = hybrid_pool_layer((2, 2))(x)

    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = hybrid_pool_layer((2, 2))(x)

    x = tf.keras.layers.Conv2D(4, kernel_size=(3, 3), strides=1, activation='relu', padding='same')(x)

    outputs = tf.keras.layers.Conv2D(4, kernel_size=(3, 3), strides=1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='hy_pool')
    model.compile(loss=tf.keras.losses.mean_squared_error,
                  # loss_weights=[0.3, 0.7],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                  metrics=['mse'])
    print(model.summary())
    return model

def _train(model, x_train, x_test, y_train, y_test, modelname):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = 'checkpoints/'+modelname+'_bestmodel.h5',
        monitor='mse',
        mode='min',
        save_best_only=True
    )
    model.fit(x_train, y_train, epochs=4500, batch_size=20, validation_data=(x_test, y_test), callbacks=[model_checkpoint_callback])
    model.save('checkpoints/'+modelname+'_model.h5')

def _predict(model):
    fig = plt.figure(figsize=(50, 50))
    for i in range(1, 10):
        print('x_test[i].shape', x_test[i].shape)
        sample_image = np.reshape(x_test[i] * 255, (96, 96)).astype(np.uint8)
        pred = model.predict(x_test[i: i + 1]) * 96
        pred = pred.astype(np.int32)
        pred = np.reshape(pred[0, 0, 0], (2, 2))

        fig.add_subplot(1, 20, i)
        plt.imshow(sample_image, cmap='gray')
        plt.scatter(pred[:, 0], pred[:, 1], s = 5, c='yellow')
    plt.show()

x_train, x_test, y_train, y_test = _load_datasets()

model1 = _hy_pool_Model_angle_mse()

_train(model1, x_train, x_test, y_train, y_test, 'U_huber_angle')

new_model1 = tf.keras.models.load_model('checkpoints/U_huber_angle_bestmodel.h5',
                                        custom_objects={'between_eyes_angle_loss': between_eyes_angle_loss,
                                                        'huber_loss': huber_loss
                                                        })

_predict(new_model1)