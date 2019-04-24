import glob
import math
import os
import numpy as np

from shutil import copyfile
from optparse import OptionParser

from src.create_image_label.create_image_label import CreateLabel
from src.unet.utils import savefile


def create_labels_from_dir(path_data, path_to, train_perc, shuffle=True):
    """
    Function that split the data as test/train and creates labels

    path_data should contrain same number of .png and .xml
    path_to is the path where train/ and test/ folder will be created

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

    split = math.floor(train_perc * nb_images)
    train_idx, test_idx = indices[:split], indices[split:]

    # Create new folders for train and test datasets
    os.mkdir(path_to + 'train/')
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

    for id in test_idx:
        filename_png = images[id].split('/')[-1]
        filename_xml = xml[id].split('/')[-1]
        if filename_png.split('.')[0] != filename_xml.split('.')[0]:
            print("The file " + str(filename_png) + " is problematic. He does not have his xml file.")
        copyfile(images[id], os.path.join(path_to + 'test/', filename_png))
        labels = CreateLabel(xml[id], images[id])
        labels = np.array(labels.get_label())
        savefile(labels, os.path.join(path_to + 'test/', filename_xml.split('.')[0]))


def get_args():
    parser = OptionParser()
    parser.add_option('--path_data', dest='path_data', type='str', default='data/raw/',
                      help='path for raw data')
    parser.add_option('--path_to', dest='path_to', type='str', default='data/',
                      help='path to export images')
    parser.add_option('--test_perc', dest='test_perc', default=0.5,
                      type='float', help='percentage of data to use as test')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
