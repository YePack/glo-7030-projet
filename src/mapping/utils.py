import os
from pathlib import PurePath
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.semantic.unet.generate_masks import create_labels_from_dir
from src.semantic.training_function import predict
from src.semantic.utils.utils import savefile, readfile


def create_mapping_data(semantic_model, path_data, path_to, train_test_perc=0.8, train_valid_perc=0.8, shuffle=True, max=None):
    create_labels_from_dir(path_data, path_to, train_test_perc, train_valid_perc, shuffle, max)
    split_folds = ['train', 'valid', 'test']
    for split_fold in split_folds:
        folder_fold = os.path.join(path_to, split_fold)
        files_predict = [f for f in os.listdir(folder_fold) if f.endswith(".png")]
        files_predict.sort()
        for file_predict in files_predict:
            output_sem, _ = predict(semantic_model, file_predict, folder_fold, after_argmax=False)
            semantic_name = PurePath(file_predict).stem + '_semantic'
            savefile(output_sem, os.path.join(folder_fold, semantic_name))


def create_projection_image(label_pickle_path, h_matrix):
    label_array = readfile(label_pickle_path)
    label_array_crop = crop_center(label_array, 450, 256)
    projected_label = warpPerspective(label_array_crop, h_matrix)
    return projected_label


def show_projected_label(projected_label):
    classes_color = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta', 'green']
    cmap = mpl.colors.ListedColormap(classes_color)
    plt.imshow(projected_label, cmap=cmap)
    plt.show()


def warpPerspective(img, M):
    mtr = img
    C, R = img.shape
    dst = np.full((C, R), 9.)
    for i in range(C):
        for j in range(R):
            res = np.dot(M, [j, i, 1])
            i2, j2, _ = (res / res[2] + 0.5).astype(int)
            if i2 >= 0 and i2 < R:
                if j2 >= 0 and j2 < C:
                    dst[j2, i2] = mtr[i, j]
    return dst


def crop_center(img, cropx, cropy):
    y, x = img.shape[0:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]

pts_src = np.array([[153, 77], [347, 163], [228, 43], [391, 100]])
pts_dst = np.array([[395, 70], [396, 189], [446, 13], [446, 143]])
h, status = cv2.findHomography(pts_src, pts_dst)
label_pickle_path = 'data/train/image-121'
new_im = create_projection_image(label_pickle_path, h)
show_projected_label(new_im)


