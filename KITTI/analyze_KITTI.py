train_gt_txt = '/home/grupo09/M5/marc/MCV_CNN_framework/KITTI/train_gt.txt'
valid_gt_txt = '/home/grupo09/M5/marc/MCV_CNN_framework/KITTI/valid_gt.txt'
test_gt_txt = '/home/grupo09/M5/marc/MCV_CNN_framework/KITTI/test_gt.txt'

label_count_train = {}
with open(train_gt_txt, 'r') as f:
    for line in f.readlines():
        label = line.strip()
        if label in label_count_train:
            label_count_train[label] += 1
        else:
            label_count_train[label] = 0

print(label_count_train, '\n')

label_count_valid = {}
for label in label_count_train.keys():
    label_count_valid[label] = 0
with open(valid_gt_txt, 'r') as f:
    for line in f.readlines():
        label = line.strip()
        label_count_valid[label] += 1

print(label_count_valid, '\n')

label_count_test = {}
for label in label_count_train.keys():
    label_count_test[label] = 0
with open(test_gt_txt, 'r') as f:
    for line in f.readlines():
        label = line.strip()
        label_count_test[label] += 1

print(label_count_test)
