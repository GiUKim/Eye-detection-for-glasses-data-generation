# Eye-detection-for-glasses-data-generation
Eye detection for glasses data generation to use dataset of object detection 
---
## Contents
### 1. Labeling 2 of eye points for training of CNN model
### 2. Compose train/val dataset
### 3. CNN model architecture and custom loss function
### 4. Putting glasses image on original face using predicted eye points
### 5. Usage(Summary)

---

## 1. Labeling 2 of eye points for training of CNN model
+ edit line 136 of [label_eye_point.py] "dataset_dir = ???" to your dataset location (the images directory path)
+ "python label_eye_point.py"
+ labelimg 2 of eye points to mouse click( must be labeled 2 points )

    <img src="https://user-images.githubusercontent.com/59654033/142639989-f8fa989f-c345-45bd-b8ee-8788b788b792.png" width="480" height="300">
    
+ if R mouse click, remove 1 label
+ if press 'a' key, move to previous image
+ if press 'd' key, move to next image

    ![image](https://user-images.githubusercontent.com/59654033/142640288-17f572a5-8c2a-4593-9829-8dd0c607691c.png)
    
    ex) x1=0.564, y1=0.503, x2=0.674, y2=0.492 ratio labeling at *.jpg <-> *.data
    
## 2. Compose train/val dataset
+ edit line 8 of [make_numpy_from_dataset.py] "dataset_dir = ???" to your dataset location (the images and labeled file directory path)
+ "python make_numpy_from_dataset.py"
+ train : val = 95 : 5 -> if want to change, edit line 9, "validation_ratio = ???"
+ x_test.npy, x_train.npy, y_test.npy, y_train.npy are saved at base directory

## 3. CNN model architecture and custom loss function
 
   ```
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
   ```
   + Used loss function : Huber loss, between eyes angle loss

   ### 1. [Huber loss reference](https://www.cantorsparadise.com/huber-loss-why-is-it-like-how-it-is-dcbe47936473)
   
   ### 2. Between eyes angle loss
   As generate glasses on face using eye detection, for naturalness, it is necessary to accurately measure the angle between the ground and the eye.
   
   This loss function's policy measure gap between prediction's eyes angle and ground truth's.
   For smoothness in model learning, the loss function was made into the following formula.
   
