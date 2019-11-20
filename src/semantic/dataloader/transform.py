import numpy as np
import torch
from torchvision import transforms


class NormalizeCropTransform:
    """
    This class object apply the appropriate transformation to tensor with the normalization apply to ImageNet
    """
    def __init__(self, normalize=True, crop=None):
        self.normalize = normalize
        self.crop = crop

    def fit(self, image_array, labels_array):

        if self.crop is not None:
            image_array = self.crop_center(image_array, self.crop[0], self.crop[1])
            labels_array = self.crop_center(labels_array, self.crop[0], self.crop[1])

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if self.normalize:
            preprocess = transforms.Compose([transforms.ToTensor(), normalize])
        else:
            preprocess = transforms.Compose([transforms.ToTensor()])

        img_tensor = preprocess(image_array)
        labels_tensor = torch.LongTensor(labels_array).unsqueeze(0)
        return img_tensor, labels_tensor

    @staticmethod
    def crop_center(img, cropx, cropy):
        y, x = img.shape[0:2]
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty+cropy, startx:startx+cropx]
