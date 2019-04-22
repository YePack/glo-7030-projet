import numpy as np
import os

from PIL import Image
from torch.utils.data import Dataset

from src.create_image_label.create_image_label import CreateLabel


def load_image(file):
    return Image.open(file)


class DataGenerator(Dataset):
    def __init__(self, imagepath, labelpath, transform):
        #  make sure label match with image
        self.transform = transform
        assert os.path.exists(imagepath), "{} not exists !".format(imagepath)
        assert os.path.exists(labelpath), "{} not exists !".format(labelpath)
        self.image = []
        self.label = []
        with open(imagepath, 'r') as f:
            for line in f:
                self.image.append(line.strip())
        with open(labelpath, 'r') as f:
            for line in f:
                self.label.append(line.strip())

    def __getitem__(self, index):
        filename = self.image[index]
        filenameGt = self.label[index]

        with open(filename, 'rb') as f:
            image_path = f
            image = np.array(load_image(f))[..., :3]
        #with open(filenameGt, 'rb') as f:
        labels_class = CreateLabel(filenameGt, filename)
            #label = load_image(f).convert('P')

        labels_array = np.array(labels_class.get_label())
        labels_array = labels_array[1:, 1:]
        img_tensor, labels_tensor = self.transform.fit(image, labels_array)


        return img_tensor, labels_tensor

    def __len__(self):
        return len(self.image)
