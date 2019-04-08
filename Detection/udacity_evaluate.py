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
import argparse
import pickle
from Mask_RCNN.export_model import export
sys.path.append('Mask_RCNN')

from Mask_RCNN.config import Config

from Mask_RCNN import visualize
from Mask_RCNN import utils
import Mask_RCNN.model as modellib
from Mask_RCNN.model import log

from udacity_detect import Detector
from udacity_config import dataset_img_ids, annotations

parser   = argparse.ArgumentParser()
parser.add_argument('--trained_model_path', type=str, default='',help='')
parser.add_argument('--in_dir', type=str, default='',help='')
parser.add_argument('--save_dir', type=str, default='',help='')
parser.add_argument('--min_confidence', type=float, default=0.1,help='')

FLAGS, unparsed = parser.parse_known_args()




if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    # -------------------------------------------------- load the model
    class_names = ['Car', 'Truck', 'Pedestrian']
    detector = Detector(FLAGS.trained_model_path, 
                        FLAGS.save_dir, 
                        FLAGS.min_confidence, 
                        class_names)

    # --------------------------------------------------- load the evlauation image indices
    k = open("eval_im_idxs.txt",'r')
    lines = k.readlines()     #i am reading lines here
    eval_im_idxs = [int(x) for x in lines]

    # --------------------------------------------------- 
    detection_results_dir = 'mAP/input/detection-results/'
    ground_truth_dir      = 'mAP/input/ground-truth/'
    images_dir            = 'mAP/input/images-optional/'

    for im_idx in eval_im_idxs:
        im_name = dataset_img_ids[im_idx]
        im_path = os.path.join(FLAGS.in_dir, str(im_name)+'.jpg')
        print(im_path)
        bounding_boxes, scores, class_ids = detector.detect(im_path, visualize_results =False)
        # -------------- save image
        image = cv2.imread(im_path)
        cv2.imwrite(os.path.join(images_dir,str(im_idx)+'.jpeg'),image)
        # -------------- save gorund truth
        labels = annotations[im_name]
        labels_file = os.path.join(ground_truth_dir, str(im_idx)+'.txt')
        file = open(labels_file, "w") 
        for label in labels:
            label = [str(e) for e in label]
            file.write(label[4]+' '+label[0]+' '+label[1]+' '+label[2]+' '+label[3]+'\n')
        file.close() 
        # -------------- save detection        
        dets_file = os.path.join(detection_results_dir, str(im_idx)+'.txt')
        file = open(dets_file, "w") 
        for b, s, c in zip(bounding_boxes, scores, class_ids):
            print(b, s, c)
            b = [str(e) for e in b]
            file.write(class_names[c-1]+' '+str(s)+' '+b[1]+' '+b[0]+' '+b[3]+' '+b[2]+'\n')
        file.close()         
        
