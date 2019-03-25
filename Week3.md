# What we have implemented     



# Code structure and explanation



# Results of the different experiments      
# Results of the different experiments  

## Camvid dataset
Networks | Accuracy | Mean IoU| Loss | 
--- | --- | --- | --- |
fcn8 (50 epoch) train | 77.57% | 70.28 |0.15 |
fcn8 (50 epoch) val |  77.39% | 66.43 | 0.21 | 
fcn8atonce (50 epoch) train | 77.06% | 69.78| 0.15 |
fcn8atonce (50 epoch) val | 76.97% | 66.49 | 0.20 |

## Evaluation fcn8atonce
Dataset | Accuracy | Mean IoU| Loss | 
--- | --- | --- | --- |
Cityscapes (17 epoch) train | 70.45% | 61.95 |0.18 |
Cityscapes (17 epoch) val |  65.42% | 56.56 | 0.23 | 
KITTI (200 epoch) train | 81.18% | 75.8| 0.10 |
KITTI (200 epoch) val | 55.80% | 48.10  | 0.90 |
synthia rand cityscapes (8  epoch) train | 57.87% | 52.20 |0.18 |
synthia rand cityscapes (8  epoch) val |  59.53% | 53.63 | 0.18 | 
pascal 2012 (20 epoch) train | 95.59% | 92.12| 0.04 |
pascal 2012 (20 epoch) val | 66.32% | 57.75% | 0.05 |


# Running the code
To use our code run the following command:

````python main.py --exp_name <EXP_NAME> --exp_folder <EXP_FOLDER> --config_file <CONFIG_FILE>````



# Level of completeness of the week 2 goals       
### Task a)
- [x] Analyze the dataset.
- [x] Calculate the accuracy on train, val, and test sets.
- [x] Evaluate different techniques in the configuration file:
### Task b)
- [x] Fully convolutional networks for semantic segmentation (Long et al. CVPR, 2015)

- [x] Another paper of free choice.
### Task c)
- [x] Select one network from the state of the art (SegnetVGG, DeepLab, ResnetFCN, ...).
- [x] Integrate the new model into the framework.
- [x] Evaluate the new model on CamVid. Train from scratch and/or fine-tune. 
### Task d)
- [x] Set-up a new experiment file to image semantic segmentation on another dataset (Cityscapes, KITTI,  Synthia, ...)

- [x] Use the FCN8 model as before.

### Task e) 
- [x] Boost the performance of your network

### Task f) 

# Slides       
[slides](https://docs.google.com/presentation/d/1gzCdiyBJP6xtvoyh1U6W2u-8NkILuakHYTKSshl_TZQ/edit?usp=sharing)

# Model weights       

