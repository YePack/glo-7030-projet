import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision import transforms
from torch import optim
from PIL import Image
from src.create_image_label.create_image_label import CreateLabel
from src.unet.unet_model import UNet
from src.unet import create_labels_from_dir

from src import train
from src.dataloader import DataGenerator
from src.dataloader import NormalizeCropTransform


#{'crowd': 0, 'ice': 1, 'board': 2, 'circlezone': 3, 'circlemid': 4, 'goal': 5, 'blue': 6, 'red': 7, 'fo': 8}
colors = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta']
cmap = mpl.colors.ListedColormap(colors)

path_img_train = 'data/raw/image_train.txt'
path_xml_train = 'data/raw/xml_train.txt'
path_img_val = 'data/raw/image_val.txt'
path_xml_val = 'data/raw/xml_val.txt'

use_gpu=True

net = UNet(3, 9)
optimizer = optim.SGD(net.parameters(),
                      lr=0.005,
                      momentum=0.9,
                      weight_decay=0.0005)

transform = NormalizeCropTransform(normalize=True, crop=(450, 256))

if use_gpu:
    net = net.cuda()

criterion = nn.CrossEntropyLoss()

train(model=net, optimizer=optimizer, imagepath_train=path_img_train, labelpath_train=path_xml_train,
      imagepath_val=path_img_val, labelpath_val=path_xml_val, n_epoch=5, batch_size=2, criterion=criterion,
      transform=transform, use_gpu=use_gpu, weight_adaptation=None)

def see_image_output(net, path_img, path_xml, transform):
    net.eval()
    data = DataGenerator(path_img, path_xml, transform=transform)
    i = 0
    while i < len(data):
        fig, subfigs = plt.subplots(2, 2)
        for j, subfig in enumerate(subfigs.reshape(-1)):
            if j % 2 == 0:
                img, label = data[i]
                img.unsqueeze_(0)
                preds = net(img)
                preds_img = preds.max(dim=1)[1]
                subfig.imshow(preds_img[0], cmap=cmap)
            else:
                subfig.imshow(label[0], cmap=cmap)
                i += 1

        plt.show()


# See the train prediction
#see_image_output(net, path_img_train, path_xml_train, transform)

# See the valid prediction
#see_image_output(net, path_img_val, path_xml_val, transform)
