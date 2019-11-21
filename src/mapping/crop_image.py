import glob
import numpy as np
from PIL import Image
from pathlib import PurePath
import matplotlib

IMAGE_PATH = 'data/train/'
filenames = glob.glob(IMAGE_PATH + '*.png')


def load_image(file):
    return Image.open(file)


def crop_center(img, cropx, cropy):
    y, x = img.shape[0:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


for filename in filenames:
    with open(filename, 'rb') as f:
        image = np.array(load_image(f))[..., :3]
        new_image = crop_center(image, 450, 256)

        pure_path = PurePath(filename)
        crop_name = pure_path.name[:-4] + '_crop.png'
        path_crop_name = str(pure_path.with_name(crop_name))

        image_save = Image.fromarray(new_image, 'RGB')
        image_save.save(path_crop_name)
