# What we have implemented     
    
## Evaluation of existing object detection network 
Dataset | Accuracy | Mean IoU| Loss | 
--- | --- | --- | --- |
MS-COCO ( epoch) train | % |  | |

# Results of the different experiments  

## MS-COCO dataset
Networks | Accuracy | Mean IoU| Loss | 
--- | --- | --- | --- |


## TsingHua-TenCent 100K dataset
Networks | Accuracy | Mean IoU| Loss | 
--- | --- | --- | --- |


## KITTI dataset
Networks | Accuracy | Mean IoU| Loss | 
--- | --- | --- | --- |


## Udacity dataset
Networks | Accuracy | Mean IoU| Loss | 
--- | --- | --- | --- |


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
python3.6 udacity_detect.py --trained_model_path /media/basem/Basem/UAB_MS/M5/udacity_model/ --image q.jpg 
```

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

