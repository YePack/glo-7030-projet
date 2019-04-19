import numpy as np
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.create_image_label.create_image_label import CreateLabel


def load_image(file):
    return Image.open(file)


class DataGenerator(Dataset):
    def __init__(self, imagepath=None, labelpath=None, transform=None):
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

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        img_tensor = preprocess(image)
        #img_tensor.unsqueeze_(0)
        labels_array = labels_array[1:, 1:]
        labels_tensor = torch.LongTensor(labels_array).unsqueeze(0)
        #if self.transform is not None:
            #image, label = self.transform(image, label)
        return img_tensor, labels_tensor

    def __len__(self):
        return len(self.image)
