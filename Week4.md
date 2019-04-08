# What we have implemented     
As part of the third part of the fourth-week project, we have trained the Mask-RCNN network with the MSCOCO and Udacity dataset. We implemented three files ourselfs to make the model work on this data:
Detection/tower_data_config.py ⇒ extend classes config() and Dataset() for loading  data and configuring the model 
Detection/tower_data_train.py ⇒ train the model and intializing weights with the pretrained weigths 
udacity_detect.py ⇒ to run the model on inference mode on images randomly picked from web

## Evaluation of existing object detection network: Trained from scratch MASK RCNN Implementation (Tensorflow)
We changed the image resolution from 1024x1024 to 512x512 to speed up the training

Split* | AP IoU 0.5-0.95 (all areas) | AP IoU 0.5 | AP IoU 0.75 | AP IoU 0.5-0.95 (small area)|AP IoU 0.5-0.95 (medium area)|AP IoU 0.5-0.95 (large area)|
--- | --- | --- | --- | --- | --- | --- |
Train | 0.050 |0.124 |0.031 | 0.018 |0.058 |0.100|
Validation | 0.044 |0.097  | 0.037 | 0.013 | 0.044|0.096|
*MSCOCO does not provide the ground truth for the test split, to evaluate on the test split the labels must be submitted to the challenge

# Results of the different experiments  

## Udacity dataset : classes are Car, Truck, Pedestrian 
--- |Class Car AP = 59,39%|Class pedrastian AP = 21,83%|Class truck AP = 28,44%|
--- | --- | --- | --- |--- |
Precision | 0.68 |0.3 |0.58|
Recall | 0.63 |0.35|0.3|

## Udacity dataset : Training Mask-RCNN 
Split* |IoU 0.2- Map = 46.05%| IoU 0.5  mAP= 36.36%|
--- | --- | --- | --- |
car | 0.7 |0.59 |
pedrastian | 0.36 |0.28|
truck | 0.33 |0.22|

# Running the code
To use our code run the following command:

```
cd Detection
```
To train on Udacity you need to download first the datatset and the model mask_rcnn_coco.h5 pretrained on COCO
```
python3.6 udacity_train.py --dataset_dir /dataset/object-detection-crowdai/ --coco_model_path ../../mask_rcnn_coco.h5 --trained_model_path /udacity_model/
```
To run the model on some image 
```
python3.6 udacity_detect.py --trained_model_path /udacity_model/ --image q.jpg 
```
# Boosting the performance
- We tuned the learning rate by tried smaller and bigger lr by factor of 10 and the best one 0.001
- We tried with model pretrained on Imagenet and COCO
- We tried different input image sizes 512, 1024
- we splitted the data into 90% for training and 10% for testing

# Level of completeness of the week 2 goals       
### Task a)
- [x] Analyze the dataset.
- [x] Calculate the mAP and FPS on train, val and test sets.
### Task b)
- [x] Faster R-CNN
- [x] Mask R-CNN
### Task c)
- [x] Train the networks on a different dataset
### Task d)
- [x] Boost the performance of your network and improve the code.

# Slides       
[slides](https://docs.google.com/presentation/d/14OWzVypZ0ZLLIrrqDhdUhpfYZmu8qmds_oiqyToUG8s/edit?usp=sharing)

# Model weights       
[weights of the trained models]()

