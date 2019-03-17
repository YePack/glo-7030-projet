import numpy as np
import os
import matplotlib.pyplot as plt
import mahotas
from src.parser.xml_parser import parse_xml_data
from collections import OrderedDict

def render(poly):
    """Return polygon as grid of points inside polygon.

    Input : poly (list of lists)
    Output : output (list of lists)
    """
    xs, ys = zip(*poly)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    newPoly = [(int(x - minx), int(y - miny)) for (x, y) in poly]

    X = maxx - minx + 1
    Y = maxy - miny + 1

    grid = np.zeros((X, Y), dtype=np.int8)
    mahotas.polygon.fill_polygon(newPoly, grid)

    return [(x + minx, y + miny) for (x, y) in zip(*np.nonzero(grid))]

path_xml = os.path.expanduser('./data/xml/test1.xml')
labels, points = parse_xml_data(path_xml)

label_to_int = {'ice': 1, 'foz': 2, 'cz': 3, 'cm': 4, 'fon': 5, 'vert': 6, 'hor':7, 'corner': 8}
frame_image = np.zeros((2875, 1609))
for i in range(len(labels)):
    poly = points[i]
    x, y = zip(*render(poly))
    for k in range(len(y)):
        if label_to_int[labels[i]] > frame_image[x[k]][y[k]]:
            frame_image[x[k]][y[k]] = label_to_int[labels[i]]


plt.imshow(frame_image.transpose())
plt.show()
plt.imsave('fig.png')