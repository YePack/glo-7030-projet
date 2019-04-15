import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms
from torch import optim
from PIL import Image
from src.create_image_label.create_image_label import CreateLabel
from src.unet.unet_model import UNet

path_img = 'data/image/resize-512x256/image3_resize_small.png'
path_xml = 'data/xml/resize-512x256/image3_512x256.xml'

img = np.array(Image.open(path_img))[..., :3]
labels = CreateLabel(path_xml, path_img)
labels = np.array(labels.get_label())

# Resize les labels pour fiter avec l'image
imgH = img.shape[0]
imgW = img.shape[1]

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.ToTensor(),
   normalize
])

img_tensor = preprocess(img)
# Vu qu'on test avec une batch_size de 1 pour le moment
img_tensor.unsqueeze_(0)
labels = np.resize(labels, (imgH, imgW))
labels_tensor = torch.from_numpy(labels)
labels_tensor.unsqueeze_(0)

net = UNet(3, 9)

# Parametres d'entrainement
optimizer = optim.SGD(net.parameters(),
                          lr=0.01,
                          momentum=0.9,
                          weight_decay=0.0005)

loss = nn.CrossEntropyLoss()
net.train()

preds = net(img_tensor)
preds = preds.max(dim=1)[1]
preds_flat = preds.view(-1, 1)
labels_flat = labels_tensor.reshape(-1, 1).long()
test1 = torch.randint(0, 8, (115456,)).reshape(-1, 1)
test = torch.randint(0, 8, (115456,)).reshape(-1, 1)
loss(preds_flat, labels_flat)