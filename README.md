# CNN framework for PyTorch
# Team members
    - Basem Elbarashy   > basemgalal.elbarashy@e-campus.uab.cat
    - Sergi Garcia Bordils  >sergi.garciab@e-campus.uab.cat
    - Marc Perez Quintana   >marc.Perez@e-campus.uab.cat
    - Gabriela Cordova >GabrielaElizabeth.Cordova@e-campus.uab.cat
# M5 Project: Scene Understanding for Autonomous Vehicles

The goal of Module 5 is to learn the basic concepts and techniques to build deep neural networks to detect, segment and recognize specific objects.
The first part of this module's project will be object recognition in a given window. We will fine-tune pre-trained networks and design new architectures. We will test our approach on a large dataset. An object recognition framework is supplied with the VGG network.

We will use deep learning frameworks such as PyTorch, Theano, TensorFlow, Caffe and Keras and basic deep learning methods such as feedforward networks (MLP) and convolutional networks (CNN). 

# Code structure and explanation

The code structured such that main components are in four directories:`config/`, `dataloader/`, `models/`, `tasks/` and they are all called from the **main.py**

**main.py** read the config file that has been specified when you run the code using Configuration() that is defined in   `config/` and based on its arguments it will do the following:     
1- Load the model using Model_builder() defined in `models/ `    
2- Select the experiment that could be detection, classification or segmentation and should be defined in `tasks/`     
3- Load the dataset using Dataloader_Builder defined in `dataloader/` and depending on the task type it will be loaded differently. For each dataset we need to have 6 files in specific format and their paths defined in the  configuration file      
4- Run training ⇒  validation ⇒ testing ⇒ prediction (doesn’t need ground truth)          
     
Other directories like `metrics/` and `utils/` provides implementations of evaluation metrics and some utility functions to the basic modules.



# Report Latex

Link to the [Overleaf article](https://www.overleaf.com/project/5c77257d723d50236d473fd9)

# VGG summary
in Very Deep Convolution Networks for Large-Scale Image Recognition, Simonyan and Zisserman analyze how increasing the depth of convolutional neural networks affects their performance. They use smaller filters (3x3, padding 1, stride 1) and argue that by stacking 3x3 convolutional layers they get a larger effective receptive field while decreasing the number of parameters. This also allows to include more non-linear rectification layers, making the decision function more discriminative.
They analyze different architectures with these 3x3 convolutional layers combined with max-pooling layers followed by 2 fully-connected layers of 4096 channels, a 1000 fully connected layer and soft-max.
They show that increasing the depth of the network helps to improve the classification accuracy and explain details of training and show how their networks perform across different datasets. 

# Inception-V3 summary
The main idea of the inception model is not only going deeper as most of the state-of-the-art models but also going wider. They did several experiments regarding the convolution factorization and used three main inception blocks to build their architecture. Each inception block has a pooling layer and in parallel other different paths of convlutional filters and output of all paths are concatenated. 1*1 conv layers used to decrease the depth and also the computational cost in many cases without hurting the performance. The pooling inside inception module is always applied in parallel with conv with stride of 2 to avoid representational bottlenecks that pooling may cause. Finally they showed that their architecture outperformed previous models by 25% reduction in the error on ILSVRC challenge.
