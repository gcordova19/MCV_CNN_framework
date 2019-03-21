from PIL import Image
import numpy as np


dataset_dir = '/home/mcv/datasets/M5/segmentation/camvid/'

train_labels = dataset_dir+'train_labels.txt'

num_images_train = 0
class_count_train = {}
for k in range(12):
    class_count_train[k] = 0

with open(train_labels, 'r') as f:
    for line in f.readlines():
        num_images_train += 1
        mask_file = line.strip()
        im_frame = Image.open(mask_file)
        np_frame = np.array(im_frame.getdata())
        unique, counts = np.unique(np_frame, return_counts=True)
        d = dict(zip(unique, counts))
        for pixel_label in d:
            class_count_train[pixel_label] += d[pixel_label]


print("class_count_train")
print(class_count_train)
print("num_images_train")
print(num_images_train)


val_labels = dataset_dir+'val_labels.txt'

num_images_val = 0
class_count_val = {}

for k in range(12):
    class_count_val[k] = 0

with open(val_labels, 'r') as f:
    for line in f.readlines():
        num_images_val += 1
        mask_file = line.strip()
        im_frame = Image.open(mask_file)
        np_frame = np.array(im_frame.getdata())
        unique, counts = np.unique(np_frame, return_counts=True)
        d = dict(zip(unique, counts))
        for pixel_label in d:
            class_count_val[pixel_label] += d[pixel_label]


print("class_count_val")
print(class_count_val)
print("num_images_val")
print(num_images_val)


test_labels = dataset_dir+'test_labels.txt'

num_images_test = 0
class_count_test = {}
for k in range(12):
    class_count_test[k] = 0

with open(test_labels, 'r') as f:
    for line in f.readlines():
        num_images_test += 1
        mask_file = line.strip()
        im_frame = Image.open(mask_file)
        np_frame = np.array(im_frame.getdata())
        unique, counts = np.unique(np_frame, return_counts=True)
        d = dict(zip(unique, counts))
        for pixel_label in d:
            class_count_test[pixel_label] += d[pixel_label]


print("class_count_test")
print(class_count_test)
print("num_images_test")
print(num_images_test)


