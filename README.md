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
