from PIL import Image
import numpy as np

im_frame = Image.open('train_mask_1TP_006690.png')
np_frame = np.array(im_frame.getdata())

unique, counts = np.unique(np_frame, return_counts=True)
d = dict(zip(unique, counts))
for k in range(12):
    if not k in d.keys():
        d[k] = 0
print(d)
