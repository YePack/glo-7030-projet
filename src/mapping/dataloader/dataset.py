from torch import LongTensor
from torch.utils.data import Dataset

from src.semantic.utils.utils import readfile


class DataGenerator(Dataset):
    def __init__(self, imagepath, labelpath):
        #  make sure label match with image
        self.image = imagepath
        self.label = labelpath

    def __getitem__(self, index):
        filename = self.image[index]
        filenameGt = self.label[index]

        image_array = readfile(filename.replace('.pkl', ''))
        label_array = readfile(filenameGt.replace('.pkl', ''))

        image_tensor = LongTensor(image_array).unsqueeze(0)
        label_array = LongTensor(label_array).unsqueeze(0)

        return image_tensor, label_array

    def __len__(self):
        return len(self.image)
