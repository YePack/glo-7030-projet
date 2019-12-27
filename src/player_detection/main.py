import glob
import os

from torch.utils.data import DataLoader

from src.player_detection.data.dataset import DataBoxesGenerator
from src.semantic.dataloader import NormalizeCropTransform


image_path = '/Users/stephanecaron/Downloads/test-xml/'
images_files = glob.glob(image_path + '*.png')
annotation_file = 'src/player_detection/data/annotations.csv'

transform = NormalizeCropTransform(normalize=True, crop=(450, 256))
data = DataBoxesGenerator(images_files, annotation_file, transform)

loader_train = DataLoader(data, batch_size=1, shuffle=True)
