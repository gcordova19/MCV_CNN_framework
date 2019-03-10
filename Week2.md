# What we have implemented     

# Code structure and explanation

The code structured such that main components are in four directories:`config/`, `dataloader/`, `models/`, `tasks/` and they are all called from the **main.py**

**main.py** read the config file that has been specified when you run the code using Configuration() that is defined in   `config/` and based on its arguments it will do the following:     
1- Load the model using Model_builder() defined in `models/ `    
2- Select the experiment that could be detection, classification or segmentation and should be defined in `tasks/`     
3- Load the dataset using Dataloader_Builder defined in `dataloader/` and depending on the task type it will be loaded differently. For each dataset we need to have 6 files in specific format and their paths defined in the  configuration file      
4- Run training ⇒  validation ⇒ testing ⇒ prediction (doesn’t need ground truth)          
     
Other directories like `metrics/` and `utils/` provides implementations of evaluation metrics and some utility functions to the basic modules.

# Results of the different experiments      

# Level of completeness of the week 2 goals       

# Slides       
[slides](https://docs.google.com/presentation/d/16mqkDaZYkFHeDiLis_u2VfJKfOyEdLi1wrwsCPWdkEE/edit?usp=sharing)

# Model weights       
[VGG trained on TT100K with pretrained weights from Imagenet](https://drive.google.com/file/d/1rzPV77QBgUsMBE7Zrk04B7wlrVOtwtJf/view?usp=sharing)

[VGG trained on KITTI with pretrained weights from Imagenet](https://drive.google.com/file/d/1om12oqCvw7WgqJEcsobZt8-ksgGtM-ms/view?usp=sharing)
