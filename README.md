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
+ ![image](https://user-images.githubusercontent.com/59654033/142639989-f8fa989f-c345-45bd-b8ee-8788b788b792.png)
+ if R mouse click, remove 1 label
+ if press 'a' key, move to previous image
+ if press 'd' key, move to next image
+ ![image](https://user-images.githubusercontent.com/59654033/142640288-17f572a5-8c2a-4593-9829-8dd0c607691c.png)

