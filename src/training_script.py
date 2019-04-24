import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision import transforms
from torch import optim
from src.unet.unet_model import UNet

from src import train
from src.dataloader import DataGenerator
from src.dataloader import NormalizeCropTransform
from src.unet.generate_masks import create_labels_from_dir


#{'crowd': 0, 'ice': 1, 'board': 2, 'circlezone': 3, 'circlemid': 4, 'goal': 5, 'blue': 6, 'red': 7, 'fo': 8}
colors = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta']
cmap = mpl.colors.ListedColormap(colors)

# Configs (use optparse instead someday)
path_data = 'data/raw/'
use_gpu=False
train_val_split=0.8

# Split train and test in 2 different folders (and save arrays instead of XMLs)
create_labels_from_dir(path_data=path_data, path_to='data/', train_perc=0.8)


# Create the network and the training stuffs
net = UNet(3, 9)
optimizer = optim.SGD(net.parameters(),
                      lr=0.005,
                      momentum=0.9,
                      weight_decay=0.0005)

transform = NormalizeCropTransform(normalize=True, crop=(450, 256))

if use_gpu:
    net = net.cuda()

criterion = nn.CrossEntropyLoss()

train(model=net, optimizer=optimizer, train_path='data/train/', n_epoch=5, train_val_split=train_val_split,
      batch_size=2, criterion=criterion, transform=transform, use_gpu=use_gpu, weight_adaptation=None)

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
see_image_output(net, train_images, path_xml_train, transform)

# See the valid prediction
#see_image_output(net, path_img_val, path_xml_val, transform)
