import numpy as np

from PIL import Image
from torch.utils.data import Dataset

from src.data_creation.file_manager import readfile


def load_image(file):
    return Image.open(file)


class DataGenerator(Dataset):
    def __init__(self, imagepath, labelpath, transform):
        #  make sure label match with image
        self.transform = transform
        self.image = imagepath
        self.label = labelpath

    def __getitem__(self, index):
        filename = self.image[index]
        filenameGt = self.label[index]

        with open(filename, 'rb') as f:
            image = np.array(load_image(f))[..., :3]

        labels_array = readfile(filenameGt.replace('.pkl', ''))
        labels_array = labels_array[1:, 1:]
        img_tensor, labels_tensor = self.transform.fit(image, labels_array)

        return img_tensor, labels_tensor

    def __len__(self):
        return len(self.image)
