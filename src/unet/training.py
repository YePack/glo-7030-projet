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
# On enleve la premiere ligne et la premiere colone car on commencait a 1
labels = labels[1:, 1:]
labels_tensor = torch.LongTensor(labels).unsqueeze(0)

net = UNet(3, 9)

# Parametres d'entrainement
optimizer = optim.SGD(net.parameters(),
                          lr=0.01,
                          momentum=0.9,
                          weight_decay=0.0005)

criterion = nn.CrossEntropyLoss()
net.train()


preds = net(img_tensor)
optimizer.zero_grad()
loss = criterion(preds, labels_tensor)
optimizer.step()

preds_img = preds.max(dim=1)[1]
plt.imshow(preds_img[0])
plt.show()
