import glob
import numpy as np

from PIL import Image
from torchvision import transforms
from src.unet.utils import readfile, savefile

path_data = 'data/train/'


images = glob.glob(path_data + '*.png')
labels = glob.glob(path_data + '*.pkl')

preprocess_flip = transforms.Compose([
        transforms.RandomHorizontalFlip(1)
])

for image in images:
    image_flip = np.array(Image.open(image))[..., :3]
    image_flip = Image.fromarray(image_flip)

    image_flip = preprocess_flip(image_flip)
    image_flip.save(image.replace('image', 'rimage'))


for label in labels:
    label_flip = readfile(label.replace('.pkl', ''))
    label_flip = Image.fromarray(label_flip)

    label_flip = preprocess_flip(label_flip)
    label_flip = np.array(label_flip)
    savefile(label_flip, label.replace('.pkl', '').replace('image', 'rimage'))
