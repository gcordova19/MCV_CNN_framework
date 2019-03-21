
import matplotlib.pyplot as plt


label_count_train = {'i1': 5, 'i4': 475, 'i5': 1044, 'il100': 91, 'il60': 337, 'il80': 196, 'io': 579, 'ip': 189, 'p10': 243, 'p11': 968, 'p12': 105, 'p19': 86, 'p23': 162, 'p26': 514, 'p27': 83, 'p3': 110, 'p5': 257, 'p6': 68, 'pg': 103, 'ph4': 82, 'ph4.5': 121, 'ph5': 72, 'pl100': 448, 'pl120': 207, 'pl20': 97, 'pl30': 373, 'pl40': 873, 'pl5': 268, 'pl50': 659, 'pl60': 525, 'pl70': 102, 'pl80': 578, 'pm20': 106, 'pm30': 74, 'pm55': 97, 'pn': 1886, 'pne': 1378, 'po': 734, 'pr40': 135, 'w13': 89, 'w32': 69, 'w55': 108, 'w57': 262, 'w59': 121, 'wo': 72}
label_count_valid = {'i1': 0, 'i4': 26, 'i5': 114, 'il100': 1, 'il60': 5, 'il80': 2, 'io': 8, 'ip': 1, 'p10': 4, 'p11': 19, 'p12': 5, 'p19': 3, 'p23': 5, 'p26': 3, 'p27': 0, 'p3': 1, 'p5': 20, 'p6': 0, 'pg': 2, 'ph4': 0, 'ph4.5': 0, 'ph5': 1, 'pl100': 6, 'pl120': 1, 'pl20': 1, 'pl30': 10, 'pl40': 27, 'pl5': 26, 'pl50': 31, 'pl60': 10, 'pl70': 0, 'pl80': 23, 'pm20': 0, 'pm30': 0, 'pm55': 0, 'pn': 146, 'pne': 210, 'po': 7, 'pr40': 0, 'w13': 8, 'w32': 1, 'w55': 0, 'w57': 8, 'w59': 5, 'wo': 0}
label_count_test = {'i1': 1, 'i4': 231, 'i5': 504, 'il100': 39, 'il60': 140, 'il80': 96, 'io': 266, 'ip': 128, 'p10': 87, 'p11': 497, 'p12': 66, 'p19': 33, 'p23': 103, 'p26': 241, 'p27': 47, 'p3': 58, 'p5': 121, 'p6': 39, 'pg': 44, 'ph4': 37, 'ph4.5': 60, 'ph5': 40, 'pl100': 216, 'pl120': 87, 'pl20': 56, 'pl30': 205, 'pl40': 445, 'pl5': 203, 'pl50': 340, 'pl60': 274, 'pl70': 44, 'pl80': 275, 'pm20': 49, 'pm30': 32, 'pm55': 38, 'pn': 966, 'pne': 660, 'po': 389, 'pr40': 63, 'w13': 31, 'w32': 34, 'w55': 60, 'w57': 122, 'w59': 60, 'wo': 38}

plt.figure(figsize=(40,20))
print(len(label_count_train))
plt.bar(range(len(label_count_train)), list(label_count_train.values()))
plt.xticks(range(len(label_count_train)), list(label_count_train.keys()), rotation='vertical')
plt.title('Class Frequency Train TT100K')
plt.savefig('TT100K/class_frequency_train_tt100k.png')

plt.figure(figsize=(40,20))
print(len(label_count_valid))
plt.bar(range(len(label_count_valid)), list(label_count_valid.values()))
plt.xticks(range(len(label_count_valid)), list(label_count_valid.keys()), rotation='vertical')
plt.title('Class Frequency Valid TT100K')
plt.savefig('TT100K/class_frequency_valid_tt100k.png')

plt.figure(figsize=(40,20))
print(len(label_count_test))
plt.bar(range(len(label_count_test)), list(label_count_test.values()))
plt.xticks(range(len(label_count_test)), list(label_count_test.keys()), rotation='vertical')
plt.title('Class Frequency Test TT100K')
plt.savefig('TT100K/class_frequency_test_tt100k.png')