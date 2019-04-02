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
from Mask_RCNN.export_model import export
import pickle
import csv

sys.path.append('Mask_RCNN')

from Mask_RCNN.config import Config
from Mask_RCNN import visualize
from Mask_RCNN import utils
import Mask_RCNN.model as modellib
from Mask_RCNN.model import log
from tower_data_config import InferenceConfig
from tower_data_config import InferenceConfig

parser   = argparse.ArgumentParser()
parser.add_argument('--trained_model_path', type=str, default='',help='')
parser.add_argument('--image', type=str, default='',help='')
parser.add_argument('--save_dir', type=str, default='',help='')
parser.add_argument('--min_confidence', type=float, default=0.9,help='')


def saveEventsToCsv(fileName,row,startNewFile = False):
    if startNewFile:
        mode = 'w'
    else:
        mode = 'a'
    with open(fileName, mode) as csvfile:
        eventWriter = csv.writer(csvfile, delimiter=',')
        eventWriter.writerow(row)

class Detector():
    """load the model and use it"""
    def __init__(self, trained_model_path, output_dir, min_confidence, class_names):
        self.model          = None
        self.min_confidence = min_confidence
        self.model          = self.load_model(trained_model_path)
        self.class_names    = class_names 
        self.output_dir     = output_dir

    def visualize_image(self, im, bounding_boxes, scores, class_ids):
        font = cv2.FONT_HERSHEY_SIMPLEX
        print( bounding_boxes, scores, class_ids)

        for b,s,i in zip(bounding_boxes, scores, class_ids):
            i = class_names[i-1]
            if i   == 'Insulator':
                label_color = (0,0,255)
            elif i == 'Davit_Arm':
                label_color = (0,255,0)
            elif i == 'Corona_Ring':
                label_color = (255,0,0)
            else:
                label_color = (0,0,0)
            s = str(s)
            cv2.putText(im,i+' '+s[0:4],(b[1],max(b[0]-50,0)), font, 1,label_color,2,cv2.LINE_AA)
            cv2.rectangle(im,(b[1],b[0]),(b[3],b[2]),label_color,3)

        return im
    
    def saveDetections(self,bounding_boxes, scores, class_ids):
        fileName = os.path.join(self.output_dir, 'detections.csv')
        row = ['class','confidence','minX', 'minY', 'maxX', 'maxY']
        saveEventsToCsv(fileName,row,startNewFile = True)
        class_counts = {c:0 for c in self.class_names}

        for b,s,i in zip(bounding_boxes, scores, class_ids):
            i = class_names[i-1]
            s = str(s)
            s = s[0:4] 
            row = [i, s, b[1],b[0],b[3],b[2]]
            saveEventsToCsv(fileName,row)
            class_counts[i] = class_counts[i] + 1

        saveEventsToCsv(fileName,[])            
        saveEventsToCsv(fileName,[])            
        row = ['class', 'count']
        saveEventsToCsv(fileName,row)

        for c in self.class_names:
            row = [c , str(class_counts[c])]
            saveEventsToCsv(fileName,row)



    def load_model(self, trained_model_path):
        inference_config = InferenceConfig()
        inference_config.DETECTION_MIN_CONFIDENCE = self.min_confidence
        inference_config.IMAGES_PER_GPU  = 1
        inference_config.BATCH_SIZE = 1
        # for one gpu: BATCH_SIZE should be equal to IMAGES_PER_GPU and also shoul equal len(images)

        model = modellib.MaskRCNN(mode="inference", 
                                  config=inference_config,
                                  model_dir=trained_model_path)
        model_path = model.find_last()[1]
        model.load_weights(model_path, by_name=True)
        return model



    def detect(self, im, visualize_results = False):
        """im could be numpy image (read with cv2) or image path"""
        if isinstance(im, str):
            im = cv2.imread(im)
        results = self.model.detect([im], verbose=1)
        r = results[0]
        if visualize_results:
            visualize.display_instances(im, r['rois'], r['masks'], r['class_ids'], 
                               ['BG']+ self.class_names, r['scores'], ax=get_ax(),
                                output_im_dir=self.output_dir)

        bounding_boxes = r['rois'].tolist()
        scores         = r['scores'].tolist()
        class_ids      = r['class_ids'].tolist()

        return bounding_boxes, scores, class_ids

    def detect_big_image(self, im, visualize_results = False):
        if isinstance(im, str):
            im = cv2.imread(im)

        height, width = im.shape[0:2]
        max_dim = 3000
        bounding_boxes = [] 
        scores    = []
        class_ids = [] 
        tl_x         = list(range(0,width,max_dim))
        tl_width     = [max_dim] * len(tl_x)
        tl_width[-1] = width - sum(tl_width[0:-1])
        
        tl_y         = list(range(0,height,max_dim))
        tl_height    = [max_dim] * len(tl_y)
        tl_height[-1]= height - sum(tl_height[0:-1])


        for x, w in zip(tl_x, tl_width):
            for y,h in zip(tl_y, tl_height):
                print(x,y)

                boxes, scrs, ids = self.detect(im[y:y+h,x:x+w], visualize_results)
                for b in boxes:
                    b = [b[0]+y, b[1]+x, b[2]+y, b[3]+x]
                    bounding_boxes.append(b)
                scores    += scrs
                class_ids += ids
        
        # ------------- save output image and detections in csv
        
        im = self.visualize_image(im, bounding_boxes, scores, class_ids)
        cv2.imwrite(os.path.join(self.output_dir,'output.jpeg'),im)
        self.saveDetections(bounding_boxes, scores, class_ids)
        

        return bounding_boxes, scores, class_ids


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    class_names = ['Corona_Ring', 'Insulator', 'Davit_Arm']
    
    detector = Detector(FLAGS.trained_model_path, 
                        FLAGS.save_dir, 
                        FLAGS.min_confidence, 
                        class_names)
    
    #detector.detect(FLAGS.image, visualize_results =True)
    detector.detect_big_image(FLAGS.image)