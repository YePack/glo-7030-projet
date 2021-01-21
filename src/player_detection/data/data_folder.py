import glob
import os
import math

import numpy as np
from shutil import copyfile

from src.player_detection.xml_parser import xml_to_csv


def create_data_folder(folder_from,
                       folder_to,
                       train_test_perc=0.8,
                       train_valid_perc=0.8,
                       shuffle=True):

    # List all png files
    images = glob.glob(folder_from + '*.png')

    # List all XML files
    xml = glob.glob(folder_from + '*.xml')

    # Split train/valid/test
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
    train_idx, valid_idx = indices_train[:split_train], indices_train[
        split_train:]


    # Create new folders for train and test datasets
    os.mkdir(folder_to + 'train/')
    os.mkdir(folder_to + 'valid/')
    os.mkdir(folder_to + 'test/')

    # Copy images files to right folders
    for id in train_idx:
        filename_png = images[id].split('/')[-1]
        filename_xml = xml[id].split('/')[-1]
        copyfile(images[id], os.path.join(folder_to, 'train', filename_png))
        copyfile(xml[id], os.path.join(folder_to, 'train', filename_xml))
    xml_to_csv(os.path.join(folder_to, 'train'),
               output_file=os.path.join(folder_to, "train.csv"))

    for id in valid_idx:
        filename_png = images[id].split('/')[-1]
        filename_xml = xml[id].split('/')[-1]
        copyfile(images[id], os.path.join(folder_to, 'valid', filename_png))
        copyfile(xml[id], os.path.join(folder_to, 'valid', filename_xml))
    xml_to_csv(os.path.join(folder_to, 'valid'),
               output_file=os.path.join(folder_to, "valid.csv"))

    for id in test_idx:
        filename_png = images[id].split('/')[-1]
        filename_xml = xml[id].split('/')[-1]
        copyfile(images[id], os.path.join(folder_to, 'test', filename_png))
        copyfile(xml[id], os.path.join(folder_to, 'test', filename_xml))
    xml_to_csv(os.path.join(folder_to, 'test'),
               output_file=os.path.join(folder_to, "test.csv"))


if __name__ == '__main__':
    create_data_folder("data/player_detection_raw/",
                       "data/player_detection/")
