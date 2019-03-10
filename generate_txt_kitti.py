from os import listdir

dataset = "/home/grupo09/mcv/datasets/M5/classification/KITTI/"
txt_dir = "/home/grupo09/M5/marc/MCV_CNN_framework/KITTI/"

ground_truth_train = ""
count_train = 0
labels = []
with open(txt_dir+"train_images.txt", "w+") as f:
    for class_dir in listdir(dataset + "train/"):
        labels.append(class_dir)
        for image in listdir(dataset + "train/" + class_dir + "/"):
            f.write(dataset + "train/" + class_dir + "/" + image+'\n')
            ground_truth_train += class_dir +"\n"
            count_train += 1

print("generated train_images.txt and train_gt.txt at", txt_dir)
print("num of train samples:", count_train)

with open(txt_dir+"train_gt.txt", "w+") as f:
    f.write(ground_truth_train)



#Since the test folder in the server is empty we split the validation folder in test and valid.
ground_truth_valid = ""
ground_truth_test = ""
valid = True
count_valid = 0
count_test = 0
with open(txt_dir+"valid_images.txt", "w+") as f_val:
    with open(txt_dir + "test_images.txt", "w+") as f_test:
        for class_dir in listdir(dataset + "valid/"):
            for image in listdir(dataset + "valid/" + class_dir + "/"):
                if valid:
                    f_val.write(dataset + "valid/" + class_dir + "/" + image+'\n')
                    ground_truth_valid += class_dir +"\n"
                    count_valid+=1
                else:
                    f_test.write(dataset + "valid/" + class_dir + "/" + image + '\n')
                    ground_truth_test += class_dir + "\n"
                    count_test+=1
                valid = not valid

with open(txt_dir+"valid_gt.txt", "w+") as f:
    f.write(ground_truth_valid)

with open(txt_dir+"train_gt.txt", "w+") as f:
    f.write(ground_truth_test)

print("generated valid_images.txt and valid_gt.txt at", txt_dir)
print("num of valid samples:", count_valid)
print("generated test_images.txt and test_gt.txt at", txt_dir)
print("num of test samples:", count_test)

print("-----------\nlabels:")
print(labels)

map_labels = {}
count_labels = 0
for label in labels:
    map_labels[label] = count_labels
    count_labels+=1

print("-----------\nmap_labels:")
print(map_labels)

print("-----------\nnum_classes:")
print(len(labels))
