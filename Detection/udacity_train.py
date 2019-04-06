import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import sys
import argparse
sys.path.append('Mask_RCNN')

from Mask_RCNN.config import Config
import Mask_RCNN.model as modellib
from Mask_RCNN.model import log

from Mask_RCNN import visualize
from Mask_RCNN import utils

from udacity_config import dataset_img_ids, annotations, DataConfig, udacityDataset

parser   = argparse.ArgumentParser()
parser.add_argument('--dataset_dir'         , type=str, default='',help='')
parser.add_argument('--coco_model_path'     , type=str, default='',help='')
parser.add_argument('--trained_model_path'  , type=str, default='',help='')


def model_train_evaluate(dataset_train_dir, dataset_eval_dir,
                         coco_model_path, trained_model_path):
    init_with        = "coco"  # imagenet, coco, or last
    train_Heads_Only = False
    num_epochs       = 5
    images_len       = len(dataset_img_ids)
    train_images_len = int(images_len * 0.9)

    indices = list(range(0,images_len))

    np.random.seed(0)
    np.random.shuffle(indices)
    
    train_im_idxs = indices[0:train_images_len]
    eval_im_idxs  = indices[train_images_len:images_len]

    file_eval_idx = open('eval_im_idxs.txt', 'w')
    
    for im_i in eval_im_idxs:
        file_eval_idx.write("%s\n" % im_i)    
    file_eval_idx.close()

    print('Training on', train_images_len,'images')
    print('Testing on', images_len - train_images_len,'images')
    #trained_model_path = os.path.join(trained_model_path, "logs")

    # Download COCO trained weights from Releases if needed
    if not os.path.exists(coco_model_path):
        utils.download_trained_weights(coco_model_path)

    config = DataConfig()
    config.display()

    # Training dataset
    dataset_train = udacityDataset()
    dataset_train.add_dataset_dir(dataset_train_dir)
    dataset_train.load_images(train_im_idxs, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()

    # Validation dataset
    dataset_val = udacityDataset()
    dataset_val.add_dataset_dir(dataset_eval_dir)
    dataset_val.load_images(eval_im_idxs, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()

    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 2)

    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        print(class_ids)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=trained_model_path)

    # Which weights to start with?

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(coco_model_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)


    if train_Heads_Only:
        layers_to_train = 'heads'
    else:
        layers_to_train = 'all'


    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=num_epochs, 
                layers=layers_to_train)



if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    model_train_evaluate(FLAGS.dataset_dir, FLAGS.dataset_dir,
                         FLAGS.coco_model_path, FLAGS.trained_model_path)
