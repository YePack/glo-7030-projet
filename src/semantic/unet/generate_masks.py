import glob
import math
import os
import numpy as np

from shutil import copyfile
from src.semantic.utils.create_image_label import CreateLabel
from src.semantic.utils.utils import savefile


def create_labels_from_dir(path_data, path_to, train_test_perc=0.8, train_valid_perc=0.8, shuffle=True, max=None):
    """
    Function that split the data as test/valid/train and creates labels

    path_data should contrain same number of .png and .xml
    path_to is the path where train/, test/ and valid/ folders will be created

    In the resulting folders, you'll find the original .png image as
    well as a pickle file, corresponding to the mask of this image

    The XML files are created using cvat tool (see labeling-tool/)
    """

    images = glob.glob(path_data + '*.png')
    xml = glob.glob(path_data + '*.xml')

    images.sort()
    xml.sort()

    if len(images) != len(xml):
        print("You don't have the same number of .png and .xml files")
        exit

    nb_images = len(images)
    indices = np.arange(nb_images)

    if shuffle:
        np.random.shuffle(indices)

    split = math.floor(train_test_perc * nb_images)
    train_idx, test_idx = indices[:split], indices[split:]

    nb_images_train = len(train_idx)
    indices_train = np.arange(nb_images_train)

    if shuffle:
        np.random.shuffle(indices_train)

    split_train = math.floor(train_valid_perc * nb_images_train)
    train_idx, valid_idx = indices_train[:split_train], indices_train[split_train:]

    if max is not None:
        train_idx = train_idx[:max]

    # Create new folders for train and test datasets
    os.mkdir(path_to + 'train/')
    os.mkdir(path_to + 'valid/')
    os.mkdir(path_to + 'test/')

    for id in train_idx:
        filename_png = images[id].split('/')[-1]
        filename_xml = xml[id].split('/')[-1]
        if filename_png.split('.')[0] != filename_xml.split('.')[0]:
            print("The file " + str(filename_png) + " is problematic. He does not have his xml file.")
        copyfile(images[id], os.path.join(path_to + 'train/', filename_png))
        labels = CreateLabel(xml[id], images[id])
        labels = np.array(labels.get_label())
        savefile(labels, os.path.join(path_to + 'train/', filename_xml.split('.')[0]))

    for id in valid_idx:
        filename_png = images[id].split('/')[-1]
        filename_xml = xml[id].split('/')[-1]
        if filename_png.split('.')[0] != filename_xml.split('.')[0]:
            print("The file " + str(filename_png) + " is problematic. He does not have his xml file.")
        copyfile(images[id], os.path.join(path_to + 'valid/', filename_png))
        labels = CreateLabel(xml[id], images[id])
        labels = np.array(labels.get_label())
        savefile(labels, os.path.join(path_to + 'valid/', filename_xml.split('.')[0]))

    for id in test_idx:
        filename_png = images[id].split('/')[-1]
        filename_xml = xml[id].split('/')[-1]
        if filename_png.split('.')[0] != filename_xml.split('.')[0]:
            print("The file " + str(filename_png) + " is problematic. He does not have his xml file.")
        copyfile(images[id], os.path.join(path_to + 'test/', filename_png))
        labels = CreateLabel(xml[id], images[id])
        labels = np.array(labels.get_label())
        savefile(labels, os.path.join(path_to + 'test/', filename_xml.split('.')[0]))

