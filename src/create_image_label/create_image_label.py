import numpy as np
import matplotlib.pyplot as plt
import mahotas
from src.parser.xml_parser import parse_xml_data
from src.net_parameters import p_label_to_int
from PIL import Image


class CreateLabel:
    def __init__(self, path_xml, path_image):
        self.path_xml = path_xml
        self.path_image = path_image
        self.frame_image = None

    @staticmethod
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

    def get_label(self):
        im = Image.open(self.path_image)
        labels, points = parse_xml_data(self.path_xml)

        frame_image = np.zeros((im.size[0]+1, im.size[1]+1))
        for i in range(len(labels)):
            poly = points[i]
            x, y = zip(*CreateLabel.render(poly))
            for k in range(len(y)):
                if p_label_to_int[labels[i]] > frame_image[x[k]][y[k]]:
                    frame_image[x[k]][y[k]] = p_label_to_int[labels[i]]
        self.frame_image = frame_image.transpose()
        return frame_image.transpose()

    def show_plot(self):
        if self.frame_image is None:
            raise Exception
        else:
            plt.imshow(self.frame_image)
            plt.show()


#Label2 = CreateLabel(path_xml='./data/xml/test2_polygon.xml', path_image='./data/image/test2_polygon.png')
#label2_array = Label2.get_label()
#Label2.show_plot()
