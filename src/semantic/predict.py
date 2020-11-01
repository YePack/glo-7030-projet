import os
import matplotlib as mpl
from optparse import OptionParser
import matplotlib.pyplot as plt

from src.semantic.training_function import predict
from src.data_creation.file_manager import readfile, savefile

from src.semantic.net_parameters import p_classes_color

cmap = mpl.colors.ListedColormap(p_classes_color)

def get_args():
    parser = OptionParser()
    parser.add_option('-p', '--path', type=str, dest='path_model', default='/Users/stephanecaron/Downloads/unet',
                      help='Path to model object')
    parser.add_option('-f', '--folder', dest='folder', default='/Users/stephanecaron/Downloads/test-video', type=str,
                      help='Folder containing images to predict')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = readfile(args.path_model)

    i=0

    files_predict = [f for f in os.listdir(args.folder) if f.endswith(".png") and f.startswith("resized_")]
    files_predict.sort()

    for file in files_predict:
        image_pred, image_raw = predict(net, file, args.folder)
        fig, subfigs = plt.subplots(2, 1)
        subfigs[0].imshow(image_raw)
        subfigs[1].imshow(image_pred[0], cmap=cmap)
        subfigs[0].axis('off')
        subfigs[1].axis('off')
        plt.savefig(os.path.join(args.folder,"prediction_"+"%04d" % i+".png"))
        print(f"saved {file} to {i}")
        i+=1
