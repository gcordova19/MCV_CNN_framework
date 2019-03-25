# What we have implemented     



# Code structure and explanation



# Results of the different experiments      
## Camvid dataset
Networks | Accuracy | Mean IoU| Loss | 
--- | --- | --- | --- |
fcn8 (50 epoch) train | 77.57% | 70.28 |0.15 |
fcn8 (50 epoch) val |  77.39% | 66.43% | 0.21 | 
fcn8atonce (50 epoch) train | 77.06% | 69.78| 0.15 |
fcn8atonce (50 epoch) val | 76.97% | 66.49% | 0.20 |



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

