from resizeimage import resizeimage
from PIL import Image


with open('./data/image/image6.png', 'r+b') as f:
    with Image.open(f) as image:
        cover = resizeimage.resize_thumbnail(image, [512, 256])
        cover.save('./data/image/image6_resize_small.png', image.format)
