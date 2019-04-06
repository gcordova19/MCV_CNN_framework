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

# Report Latex

Link to the [Overleaf article](https://www.overleaf.com/project/5c77257d723d50236d473fd9)

# Week2
[Summary of our work](Week2.md)

# Week3
[Summary of our work](Week3.md)

# Week4
[Summary of our work](Week4.md)


# VGG summary
in Very Deep Convolution Networks for Large-Scale Image Recognition, Simonyan and Zisserman analyze how increasing the depth of convolutional neural networks affects their performance. They use smaller filters (3x3, padding 1, stride 1) and argue that by stacking 3x3 convolutional layers they get a larger effective receptive field while decreasing the number of parameters. This also allows to include more non-linear rectification layers, making the decision function more discriminative.
They analyze different architectures with these 3x3 convolutional layers combined with max-pooling layers followed by 2 fully-connected layers of 4096 channels, a 1000 fully connected layer and soft-max.
They show that increasing the depth of the network helps to improve the classification accuracy and explain details of training and show how their networks perform across different datasets. 

# Inception-V3 summary
The main idea of the inception model is not only going deeper as most of the state-of-the-art models but also going wider. They did several experiments regarding the convolution factorization and used three main inception blocks to build their architecture. Each inception block has a pooling layer and in parallel other different paths of convlutional filters and output of all paths are concatenated. 1*1 conv layers used to decrease the depth and also the computational cost in many cases without hurting the performance. The pooling inside inception module is always applied in parallel with conv with stride of 2 to avoid representational bottlenecks that pooling may cause. Finally they showed that their architecture outperformed previous models by 25% reduction in the error on ILSVRC challenge.

# Fully Convolutional Networks for Semantic Segmentation
Object detection, key points predictions and local correspondence were improved by Convenets. Deep feature hierarchies encode localizations and semantics in the nonlinear local-to-global pyramid.  FCNs adapt the input to any size and the output dimension are typically reduced by subsampling.  Shift-and-stitch filter and upsampling give effective learning dense prediction. Classifiers change into FCNs converting all fully connected layers to convolutions 1X1. Image-to-image have learned with (two regimes) high effective batch size and correlated inputs. Combining fine layers and coarse layers let the model make local predictions that respect the global structure.FCNs improve accuracy on PASCAL VOC 2011-2, NYUDv2, SIFT Flow, and PASCAL-Context

# U-Net: Convolutional Networks for Biomedical Image Segmentation
U-Net builds upon the ideas introduced in the previously described FCN, where the images are compressed and then upsampled at the output, and skip connections are used between the two phases. In addition to several architectural modifications, the authors of this paper also relied on heavy data augmentation during the training phase. This helped the model reach state-of-the-art results in biomedical segmentation challenges, where the number of training samples was scarce. There are several modifications of the FCN introduced in this architecture. According to the authors, one important modification is the usage of a large number of feature channels during the upsampling process, which helped propagating context information to higher resolution layers.  The network also avoids using FC layers, which allows the input images to have different resolutions and aspect ratios. 

# Faster R-CNN
Faster R-CNN architecture has two networks: region proposal network (RPN) for generating region proposals and a network using these proposals to detect objects into BBox. The main difference so far from the previous ones is the elimination of selective search which was the bottleneck to generate region proposals. The time cost of generating region proposals is reduced due to the use of RPN. This network ranks region boxes (called anchors) and proposes objects. Predicted regions proposed has obtained from some tweaks and compromise to separate high superposed anchors foreground and lowest ones as background. Predict the region proposals is then reshaped using an RoI pooling layer (splits the input feature map into a fixed number of roughly equal regions and then apply Max-Pooling on every region). RoI pooling layer output is then used to classify the image within the proposed region (object detection) and predict the offset values for the BBox.

# Mask R-CNN
Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branch for classification and bounding box regression. The mask branch is a small FCN applied to each RoI, predicting a segmentation mask in a pixel-to-pixel manner. To accomplish this, Mask R-CNN fixes de misalignment problem caused by RoIPool in Faster R-CNN, by using a quantization-free layer, called RoIAlign, that faithfully preserves exact spatial locations. Mask R-CNN decouples mask and class prediction: they predict a binary mask for each class independently, without competition among classes, and rely on the network’s RoI classification branch to predict the category. the model runs at about 200ms per frame on a GPU, and training on COCO takes one to two days on a single 8-GPU machine


