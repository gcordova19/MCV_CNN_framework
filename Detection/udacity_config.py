import math
import random
import numpy as np
import cv2
import sys
import os
import pickle
import csv

sys.path.append('Mask_RCNN')

from config import Config
import utils
import visualize

im_size     = 512
min_box_dim = 30

udacity_classes     = ['Car', 'Truck', 'Pedestrian']
classes_of_interest = udacity_classes
dataset_img_ids    = []
annotations = {}  #image_id: list of bboxes in this format [xmin,ymin,xmax,ymax,Label]

with open('udacity_labels/labels_crowdai.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        xmin, ymin, xmax, ymax, frame, label = row[0:6]
        img_id = os.path.splitext(frame)[0]
        if not(row[0] == 'xmin'): #skip first row
            if not(img_id in dataset_img_ids):
                annotations[img_id] = []
                dataset_img_ids.append(img_id)
            annotations[img_id].append([int(xmin), int(ymin), 
                                        int(xmax), int(ymax), label])


def from_opposite_points_2_tl_br(box):
    p1x,p1y,p2x,p2y = box
    tl = [min(p1x,p2x) ,min(p1y,p2y)]
    br = [max(p1x,p2x) ,max(p1y,p2y)]
    return [tl[0], tl[1], br[0], br[1]]


def upscale_bbox(bbox, height, width):
    #bbox = [tlx, tly, brx, bry, category]
    tlx, tly, brx, bry, category = bbox

    # ============ width check
    w   = brx - tlx
    #print(w)
    if w <= min_box_dim:
        pad = int(np.ceil((min_box_dim - w)/2 + 1))
        tlx = max(tlx-pad,0)
        brx = min(brx+pad,width-1)
    # ============ hight check
    h   =  bry - tly
    #print(h)
    if h <= min_box_dim:
        pad = int(np.ceil((min_box_dim - h)/2 + 1))
        tly = max(tly-pad,0)
        bry = min(bry+pad,height-1)
    # ============

    bbox = [tlx, tly, brx, bry, category ]

    return bbox




class DataConfig(Config):
    """Configuration for training on the seat dataset.
    Derives from the base Config class and overrides values specific
    to the seat dataset.
    """
    # Give the configuration a recognizable name
    NAME = "udacity_dataset"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(classes_of_interest)  # background + 1 object

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = im_size
    IMAGE_MAX_DIM = im_size

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_SCALES = (16, 32,64, 128, 256)
    #RPN_ANCHOR_SCALES = (32,64, 128, 256)
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


class InferenceConfig(DataConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9


class udacityDataset(utils.Dataset):
    """Generates the dataset.
    """

    def add_dataset_dir(self, data_dir):
        self.data_dir = data_dir

    def load_images(self,images_idxs, height, width):
        """
        Generate the requested number of images.
        count: number of images to generate.
        """
        # Add classes
        for i in range(1,len(udacity_classes)+1):
            self.add_class("udacity_dataset", i, udacity_classes[i-1])

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of seat sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in images_idxs:
            #bg_color, seat = self.random_image(height, width)
            self.add_image("udacity_dataset", image_id=i, path=None,
                           width=width, height=height)

    def load_image(self, image_id):
        image_no = dataset_img_ids[self.image_info[image_id]['id']]
        im_dir = os.path.join(self.data_dir , str(image_no)+'.jpg')

        image = cv2.imread(im_dir)
        #print(self.image_info)
        #print(image.shape)
        self.image_info[image_id]['height'] = image.shape[0]
        self.image_info[image_id]['width' ] = image.shape[1]

        #image = cv2.resize(image, (im_size, im_size))
        
        return image


    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "udacity_dataset":
            return ""
        else:
            super(self.__class__).image_reference(self, image_id)


    def load_mask(self, image_id):
        """Generate instance masks for the given image ID.
        """
        info = self.image_info[image_id]
        image_no = dataset_img_ids[info['id']]
        labels = annotations[image_no]

        """
        labels_of_interest = []

        for l in labels:
            l[0:4] = from_opposite_points_2_tl_br( l[0:4] )
            if l[4] in classes_of_interest:
                l = upscale_bbox(l, info['height'], info['width'])
                w = l[2]-l[0]
                h = l[3]-l[1]
                if w>min_box_dim and h>min_box_dim:
                    #print(l)
                    labels_of_interest.append(l)
        labels = labels_of_interest
        """
        count = len(labels)

        mask      = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        class_ids = []
        for i, label in enumerate(labels):
            mask[label[1]:label[3],label[0]:label[2],i] = 1
            class_ids.append(self.class_names.index(label[4]))

            #cv2.imshow('msk',mask[:,:,i]*255)
            #cv2.waitKey()
        

        class_ids = np.array(class_ids, dtype=np.int32)
        #print(class_ids)
        #print(labels)        
        return mask, class_ids