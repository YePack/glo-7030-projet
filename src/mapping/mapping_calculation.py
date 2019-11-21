import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.semantic.utils.utils import readfile


def warpPerspective(img, M, dsize):
    mtr = img
    C, R = dsize
    dst = np.full((C, R), 9.)
    for i in range(C):
        for j in range(R):
            res = np.dot(M, [j, i, 1])
            i2, j2, _ = (res / res[2] + 0.5).astype(int)
            if i2 >= 0 and i2 < R:
                if j2 >= 0 and j2 < C:
                    dst[j2, i2] = mtr[i, j]
    return dst

from src.semantic.utils.create_image_label import CreateLabel
ll = CreateLabel(path_xml='data/rink/14_Hockey Rink.xml', path_image='data/rink/resized_full_rin2.png')
label = ll.get_label()

if __name__ =='__main__':
    import cv2
    colors2 = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta', 'green']
    cmap2 = mpl.colors.ListedColormap(colors2)
    colors = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta']
    cmap = mpl.colors.ListedColormap(colors)
    pts_src = np.array([[141, 131], [480, 159], [493, 630], [64, 601]])
    pts_dst = np.array([[318, 256], [534, 372], [316, 670], [73, 473]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    label1 = readfile('data/test/image-4')

    l_out2 = warpPerspective(label1, h, (256, 510))
    plt.imshow(np.floor(l_out2), cmap=cmap2)
    plt.show()

    plt.imshow(label1, cmap=cmap)
    plt.show()