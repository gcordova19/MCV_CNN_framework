from PIL import Image
import numpy as np
from matplotlib.image import imread

def parse_pixel(label):
    #http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/label_colors.txt
    if label==0:
        return [128, 128, 128]  # Sky = Silver Gray
    if label==1:
        return [128, 0, 0] # Building = Red
    if label==2:
        return [192, 192, 128] # Column_Pole = White
    if label==3:
        return [128, 64, 128] # Road = Soft Blue
    if label==4:
        return [0, 0, 192]  # Sidewalk = Strong Blue
    if label==5:
        return [128, 128, 0] # Tree = Yellow-GreenSoft Purple
    if label==6:
        return [192, 128, 128] # SignSymbol = Pink
    if label==7:
        return [64, 64, 128] # Fence = Purple-Blue
    if label==8:
        return [64, 0, 128] # Car = Strong Purple
    if label==9:
        return [64, 64, 0] # Pedestrian = Brown-Green
    if label==10:
        return [0, 128, 192] # Bicyclist = Soft Purple
    if label==11:
        return [0, 0, 0]  # Void = Black


dataset_dir = '/home/mcv/datasets/M5/segmentation/camvid/'
output_dir_images = '/home/grupo09/M5/MCV_CNN_framework/Camvid/test/images/'
output_dir_segmentations = '/home/grupo09/M5/MCV_CNN_framework/Camvid/test/segmentations/'

train_labels = dataset_dir+'test_labels.txt'


with open(train_labels, 'r') as f:
    for line in f.readlines():
        print(line)
        mask_file = line.strip()
        im_frame = Image.open(mask_file)
        np_frame = np.array(im_frame)
        segmentation_array = np.zeros([np_frame.shape[0], np_frame.shape[1], 3], dtype=np.uint8)

        for x in range(np_frame.shape[0]):
            for y in range(np_frame.shape[1]):
                segmentation_array[x, y, :] = parse_pixel(np_frame[x, y])
        segmentation_img = Image.fromarray(segmentation_array, 'RGB')
        segmentation_img.save(output_dir_segmentations+mask_file.split('/')[-1])

"""
im_frame = Image.open('train_mask_1TP_006690.png')
np_frame = np.array(im_frame)
segmentation_array = np.zeros([np_frame.shape[0], np_frame.shape[1], 3], dtype=np.uint8)

for x in range(np_frame.shape[0]):
    for y in range(np_frame.shape[1]):
        segmentation_array[x,y,:] = parse_pixel(np_frame[x,y])
segmentation_img = Image.fromarray(segmentation_array, 'RGB')
segmentation_img.save('out.png')




rgbArray = np.zeros((512,512,3), 'uint8')
rgbArray[..., 0] = r*256
rgbArray[..., 1] = g*256
rgbArray[..., 2] = b*256
img = Image.fromarray(rgbArray)
img.save('myimg.jpeg')
"""
