import os

from optparse import OptionParser
from PIL import Image
from resizeimage import resizeimage

RESIZE_FORMAT = [512, 256]


def resize_images(path):
    """Function that resize png files inside a dir"""

    for file in os.listdir(path):
        if file.endswith(".png"):
            with open(os.path.join(path, file), 'r+b') as f:
                with Image.open(f) as image:
                    cover = resizeimage.resize_thumbnail(image, RESIZE_FORMAT)
                    new_name = os.path.join('resized_'+file)
                    cover.save(os.path.join(path, new_name), image.format)
                    print(file+' has been resized and saved.')


def get_args():
    parser = OptionParser()
    parser.add_option('-p', '--path', type=str, dest='path',
                      help='Path of images to be resized.')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    resize_images(args.path)

