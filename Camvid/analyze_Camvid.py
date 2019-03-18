from PIL import Image
import numpy as np


dataset_dir = '/home/mcv/datasets/M5/segmentation/camvid/'

train_labels = dataset_dir+'train_labels.txt'

class_count_train = {}
for k in range(12):
    class_count_train[k] = 0

with open(train_labels, 'r') as f:
    for line in f.readlines():
        mask_file = line.strip()
        im_frame = Image.open(mask_file)
        np_frame = np.array(im_frame.getdata())
        unique, counts = np.unique(np_frame, return_counts=True)
        d = dict(zip(unique, counts))
        for pixel_label in d:
            class_count_train[pixel_label] += d[pixel_label]

print(class_count_train)
