import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.semantic.utils.utils import readfile
from PIL import Image

def load_image(file):
    return Image.open(file)


def warpPerspective(img, M, dsize):
    mtr = img
    C, R = dsize
    dst = np.full((C, R), 9.)
    for i in range(mtr.shape[0]):
        for j in range(mtr.shape[1]):
            res = np.dot(M, [j, i, 1])
            i2, j2, _ = (res / res[2] + 0.5).astype(int)
            if i2 >= 0 and i2 < R:
                if j2 >= 0 and j2 < C:
                    dst[j2, i2] = mtr[i, j]
    return dst

#from src.semantic.utils.create_image_label import CreateLabel
#ll = CreateLabel(path_xml='data/rink/14_Hockey Rink.xml', path_image='resized_full_rin2.png')
#label = ll.get_label()
#plt.imsave('rinklabel.png', label, cmap=cmap)

if __name__ =='__main__':
    import cv2
    colors2 = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta', 'green']
    cmap2 = mpl.colors.ListedColormap(colors2)
    colors = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta']
    cmap = mpl.colors.ListedColormap(colors)
    pts_src = np.array([[153, 77], [347, 163], [228, 43], [391, 100], [43, 89], [217, 186]])
    pts_dst = np.array([[395, 70], [395, 189], [446, 13], [446, 143], [387, 70], [385, 188]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    with open('data/train/image-121.png', 'rb') as f:
        image = np.array(load_image(f))[..., :3]
    l_out2 = cv2.warpPerspective(image, h, (510, 256))
    img_save = Image.fromarray(l_out2, 'RGB')
    img_save.save('nom.png')
