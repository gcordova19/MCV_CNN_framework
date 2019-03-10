# What we have implemented     

As part of the first part of the second week project, we have trained the VGG16 network with the TT100K, BelguimTSM and KITTI datasets using transfer learning and training from scratch.
We have also integrated a couple of new networks in the framework, the DenseNet121 (we used the torchvision implementation) and our own network, the lightweight OurNet, which uses depthwise separable convolutional neural networks.


# Code structure and explanation

The code structured such that main components are in four directories:`config/`, `dataloader/`, `models/`, `tasks/` and they are all called from the **main.py**

**main.py** read the config file that has been specified when you run the code using Configuration() that is defined in   `config/` and based on its arguments it will do the following:     
1- Load the model using Model_builder() defined in `models/ `    
2- Select the experiment that could be detection, classification or segmentation and should be defined in `tasks/`     
3- Load the dataset using Dataloader_Builder defined in `dataloader/` and depending on the task type it will be loaded differently. For each dataset we need to have 6 files in specific format and their paths defined in the  configuration file      
4- Run training ⇒  validation ⇒ testing ⇒ prediction (doesn’t need ground truth)          
     
Other directories like `metrics/` and `utils/` provides implementations of evaluation metrics and some utility functions to the basic modules.

# Results of the different experiments      

## VGG16
Dataset | Accuracy | Precision | Recall | F1 score |
--- | --- | --- | --- |--- |
TT100K train | 99.83% | 88.77% | 87.68% | 88.17% | 
TT100K val | 99.43% | 43.91% | 46.26% | 42.52% |
TT100K test | 99.74% | 86.34% | 87.37% | 86.42% |
BelgiumTSC finetuned ImageNet | 99.79% | 63.12% | 67.47% | 63.43% |
BelgiumTSC finetuned TT100K | 99.88% | 78.79% | 78.15% | 77.76% | 
Kitti finetuned ImageNet | 99.39% | 88.82% | 89.03% | 88.80% |
Kitti non-finetuned | 87.02% | 6.01% | 12.5% | 8.12%|

## DenseNet (TT100K)
Split | Accuracy | Precision | Recall | F1 score |  
--- | --- | --- | --- |--- |                                 
Val finetuned ImageNet | 99.58% | 55.05% | 53.87% | 51.83% | 
Test finetuned ImageNet | 99.75% | 87.22% | 87.64% | 87.06% |
Val non-finetuned | 96.51% | 0.43% | 2.17% | 0.72% |
Test non-finetuned | 97.41% | 0.46% | 2.25% | 0.75% |

## OurNet (TT100K)
Split | Accuracy | Precision | Recall | F1 score |
--- | --- | --- | --- |--- |
TT100K val | 96.51% | 0.43% |2.17% | 0.72% |
TT100K test | 96.21% | 0.28% | 2.17% | 0.49% |
Kitti val | 87.02% | 6.01% | 12.50% | 8.12% |
Kitti test | 87.02% | 6.01% | 12.50% | 8.12% |

# Running the code
To use our code run the following command:

``` python main.py --exp_name <EXP_NAME> --exp_folder <EXP_FOLDER> --config_file <CONFIG_FILE> ```

where ```<EXP_FOLDER>``` is the default destination of the log files and the model weights of the different experiments, ```<EXP_NAME>``` is the 
desired name of the experiment (and will be stored in the experiments folder), and ```<CONFIG_FILE>``` is the name of the configuration file, where the task,
 model, dataset and other parameters are specified. The VGG16 experiments can be found in the configuration folder (```./config/```).

The master branch does not currently contain the code for the DenseNet121 and OurNet, they can be found in the branch named sergi. The config files are also present in the configuration folder. I'll merge that branch into master when I have time to do it because there are a gorillion of merge conflicts.

# Level of completeness of the week 2 goals       
### Task a)
- [x] Analyze the dataset.
- [x] Calculate the accuracy on train, val, and test sets.
- [x] Evaluate different techniques in the configuration file:
- [x] Transfer learning to another dataset (BTS).
- [x] Understand which parts of the code are doing what you specify in the configuration file.
### Task b)
- [x] Set-up a new experiment file to discriminate among pedestrians, vehicles, and cyclists on KITTI dataset.
- [x] Train from scratch and fine-tuning with VGG16
### Task c)
- [x] Integrate a new model into the framework using an existing PyTorch implementation.
- [x] Evaluate the new model on TT100K.
- [x] Writing our own implementation
- [x] Compare fine-tuning vs. training from scratch.
### Task d)
- [x] Boost the performance of your networks 
### Task e) 
- [x] Report+slides showing the achieved results 

# Slides       
[slides](https://docs.google.com/presentation/d/16mqkDaZYkFHeDiLis_u2VfJKfOyEdLi1wrwsCPWdkEE/edit?usp=sharing)

# Model weights       
[VGG trained on TT100K with pretrained weights from Imagenet](https://drive.google.com/file/d/1rzPV77QBgUsMBE7Zrk04B7wlrVOtwtJf/view?usp=sharing)

[VGG trained on BelgiumTSC with pretrained weights from Imagenet](https://drive.google.com/drive/folders/1qjAuTzujN8r8Q_NDpGQISuVAZ-j3dn_f?usp=sharing)

[VGG trained on BelgiumTSC with pretrained weights from TT100K](https://drive.google.com/drive/folders/1ZmLlWdPCj-1tGkknJ3TyggMOBj7gIOk8?usp=sharing)

[VGG trained on KITTI with pretrained weights from Imagenet](https://drive.google.com/file/d/1om12oqCvw7WgqJEcsobZt8-ksgGtM-ms/view?usp=sharing)

[VGG trained on KITTI from scratch](https://drive.google.com/drive/folders/1uW-U3xQZJvyUmn9OlgV_mNKr52_EiANb?usp=sharing)

