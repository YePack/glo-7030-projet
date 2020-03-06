from torch import LongTensor
from torch.utils.data import Dataset

from src.semantic.utils.utils import readfile
from src.mapping.utils import crop_center


class DataGenerator(Dataset):
    def __init__(self, imagepath, labelpath):
        #  make sure label match with image
        self.image = imagepath
        self.label = labelpath

    def __getitem__(self, index):
        filename = self.image[index]
        filenameGt = self.label[index]

        image_tensor = readfile(filename.replace('.pkl', ''))
        label_array = readfile(filenameGt.replace('.pkl', ''))

        label_array_crop = crop_center(label_array, 450, 256)

        image_tensor = image_tensor[0]
        label_tensor = LongTensor(label_array_crop)

        return image_tensor, label_tensor

    def __len__(self):
        return len(self.image)
