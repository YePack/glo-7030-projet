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

from src.unet.training_function import train
from src.unet.Dataset import DataGenerator



colors = ['black', 'white', 'yellow', 'pink', 'coral', 'crimson', 'blue', 'red', 'magenta']
cmap = mpl.colors.ListedColormap(colors)

path_img_train = 'data/raw/image_train.txt'
path_xml_train = 'data/raw/xml_train.txt'
path_img_val = 'data/raw/image_val.txt'
path_xml_val = 'data/raw/xml_val.txt'

net = UNet(3, 9)
optimizer = optim.SGD(net.parameters(),
                          lr=0.01,
                          momentum=0.9,
                          weight_decay=0.0005)
weight_learn = torch.FloatTensor(np.array([2.0599, 1.5701, 2.5674, 2.5535, 2.7183, 4.7047, 2.6103, 2.7183, 4.6950]))
criterion = nn.CrossEntropyLoss()

train(model=net, optimizer=optimizer, imagepath_train=path_img_train, labelpath_train=path_xml_train,
      imagepath_val=path_img_val, labelpath_val=path_xml_val, n_epoch=30, batch_size=1, criterion=criterion)


net.eval()
#See the 2 train predictions
dd = DataGenerator(path_img_train, path_xml_train)
img1, label1 = dd[0]
img2, label2 = dd[1]
img1.unsqueeze_(0)
img2.unsqueeze_(0)
preds = net(img1)
preds_img = preds.max(dim=1)[1]
plt.imshow(preds_img[0], cmap=cmap)
plt.show()

preds = net(img2)
preds_img = preds.max(dim=1)[1]
plt.imshow(preds_img[0], cmap=cmap)
plt.show()

#See the 2 valid predictions
dd = DataGenerator(path_img_val, path_xml_val)
img3, label3 = dd[0]
img4, label4 = dd[1]
img3.unsqueeze_(0)
img4.unsqueeze_(0)

preds = net(img3)
preds_img = preds.max(dim=1)[1]
plt.imshow(preds_img[0], cmap=cmap)
plt.show()
preds = net(img4)
preds_img = preds.max(dim=1)[1]
plt.imshow(preds_img[0], cmap=cmap)
plt.show()