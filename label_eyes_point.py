from glob import glob
import os
import shutil
import cv2
import numpy as np
x_one_side_squeeze = -1
y_one_side_squeeze = -1

def padding_Resize(img, max_size):
    if img.shape == max_size:
        return img
    else:
        if img.shape[0] > max_size[0] or img.shape[1] > max_size[1]:
            shrink_start = 1.0
            shrink_step = 0.05
            height = img.shape[0]
            width = img.shape[1]
            while True:
                if width < 960 and height < 720:
                    break
                shrink_start = shrink_start - shrink_step
                height = img.shape[0] * shrink_start
                width = img.shape[1] * shrink_start
            img = cv2.resize(img, None, None, fx=shrink_start, fy=shrink_start)

        else:
            zoom_start = 1.0
            zoom_step = 0.05
            height = img.shape[0]
            width = img.shape[1]
            while True:
                if height > 720 or width > 960:
                    break
                zoom_start = zoom_start + zoom_step
                height = img.shape[0] * zoom_start
                width = img.shape[1] * zoom_start
            zoom_start = zoom_start - zoom_step
            img = cv2.resize(img, None, None, fx=zoom_start, fy=zoom_start)

        bh = int((720 - img.shape[0]) / 2)
        th = (720 - img.shape[0]) - bh
        lw = int((960 - img.shape[1]) / 2)
        rw = (960 - img.shape[1]) - lw

        # horizontal padding
        horizontal_padding_box_left = np.ones((img.shape[0], lw, 3), dtype=np.uint8)
        horizontal_padding_box_right = np.ones((img.shape[0], rw, 3), dtype=np.uint8)
        img = np.hstack((np.hstack((horizontal_padding_box_left, img)), horizontal_padding_box_right))
        x_one_side_squeeze = lw

        # vertical padding
        vertical_padding_box_top = np.ones((th, img.shape[1], 3), dtype=np.uint8)
        vertical_padding_box_bottom = np.ones((bh, img.shape[1], 3), dtype=np.uint8)
        img = np.vstack((np.vstack((vertical_padding_box_top, img)), vertical_padding_box_bottom))
        y_one_side_squeeze = th

        return img, x_one_side_squeeze, y_one_side_squeeze

def mouse_event(event, x, y, flags, param):
    i = param[0]
    x_one_side_squeeze = param[1]
    y_one_side_squeeze = param[2]
    if event == cv2.EVENT_LBUTTONDOWN:
        try:
            fr = open(imgs[idx].replace('.jpg', '.data'), 'r')
            org_label = ''
            lines = fr.readlines()
            for line in lines:
                org_label += line
            fr.close()
        except:
            org_label = ''
        f = open(imgs[idx].replace('.jpg', '.data'), 'w')
        aligned_x = x - x_one_side_squeeze
        aligned_y = y - y_one_side_squeeze
        aligned_x = round(aligned_x / (i.shape[1] - 2 * x_one_side_squeeze), 3)
        aligned_y = round(aligned_y / (i.shape[0] - 2 * y_one_side_squeeze), 3)
        org_label += str(aligned_x) + ' ' + str(aligned_y) + '\n'
        f.write(org_label)
        f.close()
        ni = cv2.imread(imgs[idx])
        try:
            coord_list = []
            f = open(imgs[idx].replace('.jpg', '.data'), 'r')
            lines = f.readlines()
            for line in lines:
                x_pt = int(float(line.split(' ')[0]) * ni.shape[1])
                y_pt = int(float(line.split(' ')[-1].split('\n')[0]) * ni.shape[0])
                coord_list.append([x_pt, y_pt])
        except:
            coord_list = [[-1, -1]]


        if [-1, -1] in coord_list:
            pass
        else:
            for coord in coord_list:
                cv2.circle(ni, (coord[0], coord[1]), 1, (0, 255, 0), -1)
        ni, x_one_side_squeeze, y_one_side_squeeze = padding_Resize(ni, WINDOW_SIZE)
        cv2.imshow('label', ni)

    elif event == cv2.EVENT_RBUTTONDOWN:
        f = open(imgs[idx].replace('.jpg', '.data'), 'r')
        lines = f.readlines()
        idx_tmp = 0
        new_label = ''
        for line in lines:
            if len(lines) - 1 == idx_tmp:
                break
            else:
                new_label += line
            idx_tmp += 1
        f.close()
        f = open(imgs[idx].replace('.jpg', '.data'), 'w')
        f.write(new_label)
        ni = cv2.imread(imgs[idx])
        try:
            coord_list = []
            f = open(imgs[idx].replace('.jpg', '.data'), 'r')
            lines = f.readlines()
            for line in lines:
                x_pt = int(float(line.split(' ')[0]) * ni.shape[1])
                y_pt = int(float(line.split(' ')[-1].split('\n')[0]) * ni.shape[0])
                coord_list.append([x_pt, y_pt])
        except:
            coord_list = [[-1, -1]]

        if [-1, -1] in coord_list:
            pass
        else:
            for coord in coord_list:
                cv2.circle(ni, (coord[0], coord[1]), 1, (0, 255, 0), -1)
        ni, x_one_side_squeeze, y_one_side_squeeze = padding_Resize(ni, WINDOW_SIZE)
        cv2.imshow('label', ni)

dataset_dir = r'C:\lab5\face_landmarks_cleaned\person40000_closed_labeled'
imgs = glob(dataset_dir + '/*.jpg')
WINDOW_SIZE = (720, 960)

idx = 0 # 3040
while True:
    i = cv2.imread(imgs[idx])
    i_fake = i.copy()
    try:
        coord_list = []
        f = open(imgs[idx].replace('.jpg', '.data'), 'r')
        lines = f.readlines()
        for line in lines:
            x_pt = int(float(line.split(' ')[0]) * i.shape[1])
            y_pt = int(float(line.split(' ')[-1].split('\n')[0]) * i.shape[0])
            coord_list.append([x_pt, y_pt])
    except:
        coord_list = [[-1, -1]]
    if [-1, -1] in coord_list:
        pass
    else:
        for coord in coord_list:
            cv2.circle(i, (coord[0], coord[1]), 1, (0, 255, 0), -1)
    i, x_one_side_squeeze, y_one_side_squeeze = padding_Resize(i, WINDOW_SIZE)

    cv2.imshow('label', i)
    cv2.setMouseCallback('label', mouse_event, param=[i, x_one_side_squeeze, y_one_side_squeeze])

    key = cv2.waitKey()
    if key == ord('d'):
        if idx == len(imgs) - 1:
            print('LAST IMAGE')
            idx = 0
        else:

            idx += 1
            print(f'{idx + 1} / {len(imgs)} : {imgs[idx]}')
        ni = cv2.imread(imgs[idx])
        try:
            coord_list = []
            f = open(imgs[idx].replace('.jpg', '.data'), 'r')
            lines = f.readlines()
            for line in lines:
                x_pt = int(float(line.split(' ')[0]) * ni.shape[1])
                y_pt = int(float(line.split(' ')[-1].split('\n')[0]) * ni.shape[0])
                coord_list.append([x_pt, y_pt])
        except:
            coord_list = [[-1, -1]]

        if [-1, -1] in coord_list:
            pass
        else:
            for coord in coord_list:
                cv2.circle(ni, (coord[0], coord[1]), 1, (0, 255, 0), -1)
        ni, x_one_side_squeeze, y_one_side_squeeze = padding_Resize(ni, WINDOW_SIZE)
        cv2.imshow('label', ni)
    elif key == ord('a'):
        if idx == 0:
            print('FIRST IMAGE')
            pass
        else:

            idx -= 1
            print(f'{idx + 1} / {len(imgs)} : {imgs[idx]}')
        ni = cv2.imread(imgs[idx])
        try:
            coord_list = []
            f = open(imgs[idx].replace('.jpg', '.data'), 'r')
            lines = f.readlines()
            for line in lines:
                x_pt = int(float(line.split(' ')[0]) * ni.shape[1])
                y_pt = int(float(line.split(' ')[-1].split('\n')[0]) * ni.shape[0])
                coord_list.append([x_pt, y_pt])
        except:
            coord_list = [[-1, -1]]

        if [-1, -1] in coord_list:
            pass
        else:
            for coord in coord_list:
                cv2.circle(ni, (coord[0], coord[1]), 1, (0, 255, 0), -1)
        ni, x_one_side_squeeze, y_one_side_squeeze = padding_Resize(ni, WINDOW_SIZE)
        cv2.imshow('label', ni)
