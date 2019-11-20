import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset
import torch
from src.utils.create_image_label import CreateLabel
from src.unet.utils import readfile


def load_image(file):
    return Image.open(file)


class DataGenerator(Dataset):
    def __init__(self, imagepath, labelpath, transform, label_prop_path=None):
        #  make sure label match with image
        self.transform = transform
        #assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        #assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        self.image = imagepath
        self.label = labelpath
        self.label_prop = label_prop_path

    def __getitem__(self, index):
        filename = self.image[index]
        filenameGt = self.label[index]
        if self.label_prop is not None:
            filenameProp = self.label_prop[index]
            labels_prop_array = readfile(filenameProp.replace('.pkl', ''))
            labels_prop_tensor = torch.FloatTensor(labels_prop_array).unsqueeze(0)
        else:
            labels_prop_tensor = torch.FloatTensor(range(9)).unsqueeze(0)

        with open(filename, 'rb') as f:
            image = np.array(load_image(f))[..., :3]

        labels_array = readfile(filenameGt.replace('.pkl', ''))
        labels_array = labels_array[1:, 1:]

        img_tensor, labels_tensor = self.transform.fit(image, labels_array)


        return img_tensor, labels_tensor, labels_prop_tensor

    def __len__(self):
        return len(self.image)
