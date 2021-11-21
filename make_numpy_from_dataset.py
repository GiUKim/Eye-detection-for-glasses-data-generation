from glob import glob
import cv2
import shutil
import os
import numpy as np
import random

dataset_dir = r'C:\lab5\face_landmarks_cleaned\eye_point_dataset'
validation_ratio = 5 # 5% validation set

imgs = glob(dataset_dir + '/*.jpg')
val_imgs = random.sample(imgs, len(imgs) // (100 // validation_ratio))
train_imgs = list(set(imgs) - set(val_imgs))

x_train = []
y_train = []
for img in train_imgs:
    i = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    i = cv2.resize(i, (96, 96))
    i = np.reshape(i, (96, 96, 1))
    x_train.append(i)
    f = open(img.replace('.jpg', '.txt'), 'r')
    lines = f.readlines()
    sub_y_train = []
    for line in lines:
        xptr = float(line.split(' ')[0])
        yptr = float(line.split(' ')[-1].split('\n')[0])
        xptr = xptr * 96
        yptr = yptr * 96
        sub_y_train.append(xptr)
        sub_y_train.append(yptr)
    y_train.append(sub_y_train)
x_train = np.array(x_train, dtype=np.float64)
y_train = np.array(y_train)

np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train )

x_test = []
y_test = []
for img in val_imgs:
    i = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    i = cv2.resize(i, (96, 96))
    i = np.reshape(i, (96, 96, 1))
    x_test.append(i)
    f = open(img.replace('.jpg', '.txt'), 'r')
    lines = f.readlines()
    sub_y_test = []
    for line in lines:
        xptr = float(line.split(' ')[0])
        yptr = float(line.split(' ')[-1].split('\n')[0])
        xptr = xptr * 96
        yptr = yptr * 96
        sub_y_test.append(xptr)
        sub_y_test.append(yptr)
    y_test.append(sub_y_test)

x_test = np.array(x_test, dtype=np.float64)
y_test = np.array(y_test)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)

print('save x_train successfully')
print('save y_train successfully')
print('save x_test successfully')
print('save y_test successfully')


