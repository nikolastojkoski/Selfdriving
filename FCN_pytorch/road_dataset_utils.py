from __future__ import print_function
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import random
import os
import re


root_dir = "../road_dataset/"
data_dir = root_dir + 'src/'
label_dir = root_dir + 'labels/'
label_colors_file = root_dir + 'label_colors.txt'
val_label_file = root_dir + 'val.csv'
train_label_file = root_dir + 'train.csv'

# # create dir for label index
label_idx_dir = os.path.join(root_dir, "Labeled_idx")
label_idx_dir = root_dir + 'Labeled_idx/'
# if not os.path.exists(label_idx_dir):
#     os.makedirs(label_idx_dir)

label2color = {}
color2label = {}
label2index = {}
index2label = {}

def divide_train_val(val_rate=0.1, shuffle=True, random_seed=None):
    data_list   = os.listdir(data_dir)
    data_len    = len(data_list)
    val_len     = int(data_len * val_rate)

    if random_seed:
        random.seed(random_seed)

    if shuffle:
        data_idx = random.sample(range(data_len), data_len)
    else:
        data_idx = list(range(data_len))

    val_idx     = [data_list[i] for i in data_idx[:val_len]]
    train_idx   = [data_list[i] for i in data_idx[val_len:]]

    # create val.csv
    v = open(val_label_file, "w")
    v.write("img,label\n")
    for idx, name in enumerate(val_idx):
        if 'png' not in name:
            continue

        # extract image idx
        regex = re.compile(r'\d+')
        image_idx = regex.findall(name)[0]
        img_name = data_dir + name
        lab_name = label_idx_dir + 'road_label_{}.png.npy'.format(image_idx)
        v.write("{},{}\n".format(img_name, lab_name))

    # create train.csv
    t = open(train_label_file, "w")
    t.write("img,label\n")
    for idx, name in enumerate(train_idx):
        if 'png' not in name:
            continue

        # extract image idx
        regex = re.compile(r'\d+')
        image_idx = regex.findall(name)[0]
        img_name = data_dir + name
        lab_name = label_idx_dir + 'road_label_{}.png.npy'.format(image_idx)

        t.write("{},{}\n".format(img_name, lab_name))


def parse_label():
    # change label to class index
    f = open(label_colors_file, "r").read().split("\n")[:-1]  # ignore the last empty line
    for idx, line in enumerate(f):
        label = line.split()[-1]
        color = tuple([int(x) for x in line.split()[:-1]])
        print(label, color)
        label2color[label] = color
        color2label[color] = label
        label2index[label] = idx
        index2label[idx]   = label
        # rgb = np.zeros((255, 255, 3), dtype=np.uint8)
        # rgb[..., 0] = color[0]
        # rgb[..., 1] = color[1]
        # rgb[..., 2] = color[2]
        # imshow(rgb, title=label)

        print('label = {}'.format(label))
        print('color = {}'.format(color))
        print('label2color = {}'.format(label2color))
        print('label2index = {}'.format(label2index))
        print('label2label = {}'.format(index2label))

    for idx, name in enumerate(os.listdir(label_dir)):
        filename = os.path.join(label_idx_dir, name)
        if os.path.exists(filename + '.npy'):
            print("Skip %s" % (name))
            continue
        print("Parse %s" % (name))
        img = os.path.join(label_dir, name)
        img = scipy.misc.imread(img, mode='RGB')
        #img = plt.imread(img)
        height, weight, _ = img.shape
    
        idx_mat = np.zeros((height, weight))
        for h in range(height):
            for w in range(weight):
                color = tuple(img[h, w])
                try:
                    label = color2label[color]
                    index = label2index[label]
                    idx_mat[h, w] = index
                except:
                    print("error: img:%s, h:%d, w:%d" % (name, h, w))
        idx_mat = idx_mat.astype(np.uint8)
        np.save(filename, idx_mat)
        print("Finish %s" % (name))

    # test some pixels' label
    img = os.path.join(label_dir, os.listdir(label_dir)[0])
    img = scipy.misc.imread(img, mode='RGB')
    test_cases = [(320, 480), (50, 50), (50, 700), (600, 455)]
    test_ans   = ['1', '0', '0', '1']
    for idx, t in enumerate(test_cases):
        color = img[t]
        assert color2label[tuple(color)] == test_ans[idx]


'''debug function'''
def imshow(img, title=None):
    try:
        img = mpimg.imread(img)
        imgplot = plt.imshow(img)
    except:
        plt.imshow(img, interpolation='nearest')

    if title is not None:
        plt.title(title)
    
    plt.show()


if __name__ == '__main__':
    divide_train_val(random_seed=1)
    parse_label()
