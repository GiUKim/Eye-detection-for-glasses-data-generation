import numpy as np
import cv2
from glob import glob
from math import atan2, degrees, tan, sin
import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
from tensorflow.keras import backend as K
import shutil

def convert(x1, x2, y1, y2, height, width):
    dw = 1./width
    dh = 1./height
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return round(x, 6), round(y, 6), round(w, 6), round(h, 6)

def angle_between(p1, p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return degrees(atan2(yDiff, xDiff))

def forward_image_and_get_label_info(model, org_img):
    h, w, _ = org_img.shape
    img = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (96, 96))
    img = np.reshape(img, (96, 96, 1))
    img = img / 255.
    pred = model.predict(img[np.newaxis, :, :])
    #pred = pred.astype(np.int32)
    pred = np.reshape(pred[0, 0, 0], (2, 2))
    eye1 = list(pred[:, 0])
    eye2 = list(pred[:, 1])
    x1 = round(eye1[0], 3)
    y1 = round(eye2[0], 3)
    x2 = round(eye1[1], 3)
    y2 = round(eye2[1], 3)
    new_label_info = str(x1) + ' ' + str(y1) + '\n' + str(x2) + ' ' + str(y2) + '\n'
    return new_label_info

def huber_loss(y_true, y_pred):
    threshold = 1.
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold

    small_error_loss = tf.square(error) / 2 # MSE
    big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold)) # MAE

    return tf.where(is_small_error, small_error_loss, big_error_loss)

def between_eyes_angle_loss(y_true, y_pred):
    # get GT angle
    x1_gt = y_true[:, :, :, 0:1]
    x2_gt = y_true[:, :, :, 1:2]
    y1_gt = y_true[:, :, :, 2:3]
    y2_gt = y_true[:, :, :, 3:4]
    x1_gt = K.reshape(x1_gt, len(x1_gt))
    x2_gt = K.reshape(x2_gt, len(x2_gt))
    y1_gt = K.reshape(y1_gt, len(y1_gt))
    y2_gt = K.reshape(y2_gt, len(y2_gt))
    # get gt angle
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

    gradient = tf.math.divide(tf.math.subtract(y2_pred, y1_pred), tf.math.subtract(x2_pred, x1_pred))
    angle_pred = tf.math.atan(gradient)
    #K.print_tensor(angle_pred, message='pred arc tan = ')
    angle_gap = tf.math.abs(tf.math.subtract(angle_gt, angle_pred))
    loss = tf.math.sin(angle_gap)
    loss = tf.math.divide(loss, 10.) # MSE 단위 밸런싱?

    loss = K.reshape(loss, (len(y2_pred), 1))
    return loss

def get_put_glasses_image_and_label_info(left_eye, right_eye, i):
    gn = random.choice(glasses_dir)
    glasses = cv2.imread(gn, cv2.IMREAD_UNCHANGED)
    print('Glasses Name:', gn.split('\\')[-1])
    org_glasses_h, org_glasses_w, _ = glasses.shape
    glasses = cv2.cvtColor(glasses, cv2.COLOR_RGB2RGBA)
    h, w, _ = i.shape
    left_eye = np.array(left_eye)
    right_eye = np.array(right_eye)
    glasses_center = np.mean([left_eye, right_eye], axis=0).astype(int)
    glasses_size = int(np.linalg.norm(left_eye - right_eye) * 2.)
    glasses_resized = cv2.resize(glasses.copy(), (glasses_size, glasses_size))

    if left_eye[0] > right_eye[0]:
        temp_eye = left_eye.copy()
        left_eye = right_eye.copy()
        right_eye = temp_eye.copy()
    angle = -angle_between(left_eye, right_eye)

    M = cv2.getRotationMatrix2D((glasses_resized.shape[1] / 2, glasses_resized.shape[0] / 2), angle, 1)
    rotated_glasses = cv2.warpAffine(glasses_resized.copy(), M, (glasses_size, glasses_size),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=255
                                     )
    x_offset, y_offset = glasses_center - np.array([rotated_glasses.shape[1] / 2,
                                                    rotated_glasses.shape[0] / 2], dtype=int)
    glasses_y_index_offset = 2
    y1, y2 = y_offset + glasses_y_index_offset, y_offset + glasses_size + glasses_y_index_offset
    x1, x2 = x_offset, x_offset + glasses_size
    alpha_s = rotated_glasses[:, :, 3] / 255.
    alpha_l = 1.0 - alpha_s
    img_result = i.copy()
    try:
        for c in range(0, 3):
            img_result[y1:y2, x1:x2, c] = (
                        alpha_s * rotated_glasses[:, :, c] + alpha_l * img_result[y1:y2, x1:x2, c])
    except:
        return None

    new_y1 = int((y2 + 2 * y1) / 3 - 2)
    new_y2 = int((y2 + 2 * y1) / 3 - 2) + int((y2 - y1) / 3) + 2
    add_height = int(abs(sin(math.pi * (angle / 180))) * (x2 - x1)) // 2
    new_x1 = x1
    new_y1 = new_y1 - add_height
    new_x2 = x2
    new_y2 = new_y2 + add_height + 2

    X, Y, W, H = convert(new_x1, new_x2, new_y1, new_y2, h, w)
    class_index = '2'
    new_label_line = class_index + ' ' + str(X) + ' ' + str(Y) + ' ' + str(W) + ' ' + str(H) + '\n'

    return img_result, new_label_line

def load_model(model_path):
    model = tf.keras.models.load_model(model_path,
                                            custom_objects={'between_eyes_angle_loss': between_eyes_angle_loss,
                                                            'huber_loss': huber_loss
                                                            }
                                            )
    return model

def put_glasses_already_exist_eye_labels(img_path):
    txt = open(img_path.replace('.jpg', '.data'), 'r')
    lines = txt.readlines()
    i = cv2.imread(img_path)
    h, w, _ = i.shape
    left_eye = []
    right_eye = []
    x1 = int(float(lines[0].split(' ')[0]) * w)
    left_eye.append(x1)
    y1 = int(float(lines[0].split(' ')[-1].split('\n')[0]) * h)
    left_eye.append(y1)
    x2 = int(float(lines[1].split(' ')[0]) * w)
    right_eye.append(x2)
    y2 = int(float(lines[1].split(' ')[-1].split('\n')[0]) * h)
    right_eye.append(y2)
    try:
        new_img, label = get_put_glasses_image_and_label_info(left_eye, right_eye, i)
        return new_img, label
    except:
        return None, None

def put_glasses_forwarding_model(img_path):
    i = cv2.imread(img_path)
    eyes_label_info = forward_image_and_get_label_info(model, i)
    f = open(img_path.replace('.jpg', '.data'), 'w')
    f.write(eyes_label_info)
    f.close()
    txt = open(img_path.replace('.jpg', '.data'), 'r')
    lines = txt.readlines()
    h, w, _ = i.shape
    left_eye = []
    right_eye = []
    x1 = int(float(lines[0].split(' ')[0]) * w)
    left_eye.append(x1)
    y1 = int(float(lines[0].split(' ')[-1].split('\n')[0]) * h)
    left_eye.append(y1)
    x2 = int(float(lines[1].split(' ')[0]) * w)
    right_eye.append(x2)
    y2 = int(float(lines[1].split(' ')[-1].split('\n')[0]) * h)
    right_eye.append(y2)
    try:
        new_img, label = get_put_glasses_image_and_label_info(left_eye, right_eye, i)
        return new_img, label
    except:
        return None, None

model = load_model('checkpoints/U_huber_angle_bestmodel.h5')
dataset_dir = r'C:\lab5\face_landmarks_cleaned\sampletest'
glasses_dir = r'C:\lab5\face_landmarks_cleaned\glasses'
result_dir = r'C:\lab5\face_landmarks_cleaned\sampletest\result'

imgs = glob(dataset_dir + '/*.jpg')
glasses_dir = glob(glasses_dir + '/*.png')
label = ''

for img in tqdm(imgs):
    if os.path.isfile(img.replace('.jpg', '.data')):
        new_img, label = put_glasses_already_exist_eye_labels(img)
        if label is None:
            continue
    else:
        # forward model
        new_img, label = put_glasses_forwarding_model(img)
        if label is None:
            continue
    cv2.imwrite(os.path.join(result_dir, img.split('\\')[-1]), new_img)
    if os.path.isfile(img.replace('.jpg', '.txt')):
        shutil.copy(img.replace('.jpg', '.txt'), result_dir)
    f = open(os.path.join(result_dir, img.split('\\')[-1].replace('.jpg', '.txt')), 'a+')
    f.write(label)
