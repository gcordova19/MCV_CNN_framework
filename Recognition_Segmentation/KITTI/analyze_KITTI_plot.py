
import matplotlib.pyplot as plt


label_count_train = {'Car': 23075, 'Pedestrian': 3636, 'Person_sitting': 172, 'Truck': 875, 'Van': 2366, 'background': 28188, 'Cyclist': 1336, 'Tram': 404}
label_count_valid = {'Car': 2833, 'Pedestrian': 425, 'Person_sitting': 25, 'Truck': 109, 'Van': 273, 'background': 3575, 'Cyclist': 145, 'Tram': 53}
label_count_test = {'Car': 2833, 'Pedestrian': 425, 'Person_sitting': 24, 'Truck': 109, 'Van': 274, 'background': 3575, 'Cyclist': 145, 'Tram': 53}

plt.figure(figsize=(15,10))
print(len(label_count_train))
plt.bar(range(len(label_count_train)), list(label_count_train.values()))
plt.xticks(range(len(label_count_train)), list(label_count_train.keys()), rotation='vertical')
plt.title('Class Frequency Train KITTI')
plt.savefig('class_frequency_train_KITTI.png')

plt.figure(figsize=(15,10))
print(len(label_count_valid))
plt.bar(range(len(label_count_valid)), list(label_count_valid.values()))
plt.xticks(range(len(label_count_valid)), list(label_count_valid.keys()), rotation='vertical')
plt.title('Class Frequency Valid KITTI')
plt.savefig('class_frequency_valid_KITTI.png')

plt.figure(figsize=(15,10))
print(len(label_count_test))
plt.bar(range(len(label_count_test)), list(label_count_test.values()))
plt.xticks(range(len(label_count_test)), list(label_count_test.keys()), rotation='vertical')
plt.title('Class Frequency Test KITTI')
plt.savefig('class_frequency_test_KITTI.png')