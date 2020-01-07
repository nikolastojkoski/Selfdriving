import cv2
import numpy as np
from os import listdir, rename
from os.path import isfile, join
import re
from FCN_pytorch.python.utils.image_labeler import PolygonDrawer

def get_dataset_mean(img_dir='../road_dataset/src/', img_name='road_src_{}.png', idx_range=range(0, 246)):
    """Calculates mean pixel value in the database for each channel R, G, B"""
    n = len(idx_range)
    pixelSum = np.zeros((640, 960, 3))
    for idx in idx_range:
        img = cv2.imread(img_dir + img_name.format(idx))
        pixelSum = np.add(pixelSum, img)
    pixelSum /= n
    means = [np.average(pixelSum[:, :, ch]) for ch in range(3)]
    print('means [B, G, R] = {}'.format(means))

def interactive_label_images(inp_folder_dir, out_folder_dir, idxLeft, idxRight,
                 src_filename_template='img_{}.png', label_filename_template='label_{}.png'):

    pd = PolygonDrawer('Labeling')
    for idx in range(idxLeft, idxRight):
        inp_img = cv2.imread(inp_folder_dir + src_filename_template.format(idx))
        label_img = pd.run(inp_img)
        cv2.imwrite(out_folder_dir + label_filename_template.format(idx), label_img)
        print('saved file: ' + out_folder_dir + label_filename_template.format(idx))


def crop_images(inp_folder_dir, out_folder_dir, idxLeft, idxRight, filename_template='img_{}.png'):

    for idx in range(idxLeft, idxRight):
        inp_img = cv2.imread(inp_folder_dir + filename_template.format(idx))

        # resize to 2482x752
        h, w, _ = inp_img.shape
        inp_img = cv2.resize(inp_img, (2 * w, 2 * h))

        # crop 960x640 region
        w_start, h_start = 820, 100
        image = inp_img[h_start: h_start + 640, w_start: w_start + 960]

        cv2.imwrite(out_folder_dir + filename_template.format(idx), image)

def threshold_labels(label_dir='../road_dataset/labels/', label_name='road_label_{}.png',
                 idx_range=range(0, 246), out_thresh_label_dir='../road_dataset/labels_fixed/'):
    for idx in idx_range:
        label = cv2.imread(label_dir + label_name.format(idx))

        for ch in range(3):
            indices = label[:,:,ch] > 0
            label[indices, 0] = label[indices, 1] = label[indices, 2] = 255

        cv2.imwrite(out_thresh_label_dir + label_name.format(idx), label)
        if idx % 10 == 0:
            print(idx)

def renumber_files_in_folder(folder_path, out_filename_template='img_{}.png',
                             starting_idx=0, safe_rename=True):

    if safe_rename == True:
        renumber_files_in_folder(folder_path, out_filename_template='tmp_{}.png', safe_rename=False)

    filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    idx = starting_idx
    for filename in filenames:
        rename(folder_path + filename, folder_path + out_filename_template.format(idx))
        idx += 1

def renumber_src_label_pairs_in_folder(folder_path,
                                       inp_src_filename='src_{}.png',
                                       inp_label_filename='label{}.png',
                                       out_src_filename='src_{}.png',
                                       out_label_filename='label_{}.png',
                                       starting_idx=0, leading_zeros=1):

    filenames = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    idx = starting_idx
    for filename in filenames:

        regex = re.compile(r'\d+')
        inp_file_idx = regex.findall(filename)[0]
        inp_file_idx = str(inp_file_idx).zfill(leading_zeros)

        rename(folder_path + inp_src_filename.format(inp_file_idx),
               folder_path + out_src_filename.format(idx))
        rename(folder_path + inp_label_filename.format(inp_file_idx),
               folder_path + out_label_filename.format(idx))

        idx += 1
