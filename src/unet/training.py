import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision import transforms
from torch import optim
from PIL import Image
from src.create_image_label.create_image_label import CreateLabel
from src.unet.unet_model import UNet


path_img = 'data/raw/image-2.png'
path_xml = 'data/raw/image-2.xml'

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
labels_tensor_array = np.array(labels_tensor.data[0])
weight_learn = torch.FloatTensor(np.array([np.exp(1-(labels_tensor_array == i).mean()) for i in range(9)]))
weight_learn[2] = weight_learn[2] + 10
# Parametres d'entrainement
optimizer = optim.SGD(net.parameters(),
                          lr=0.01,
                          momentum=0.9,
                          weight_decay=0.0005)

criterion = nn.CrossEntropyLoss(weight=weight_learn)
net.train()


for i in range(10):
    preds = net(img_tensor)
    optimizer.zero_grad()
    loss = criterion(preds, labels_tensor)
    print('iter {} loss : {}'.format(i, loss))
    loss.backward()
    optimizer.step()

preds_img = preds.max(dim=1)[1]
plt.imshow(preds_img[0])
plt.show()
preds_img.unique()
preds_softmax = F.softmax(preds, dim=1)
preds_array = np.array(preds_softmax.data[0])
preds_array.shape
labels_tensor_array = np.array(labels_tensor.data[0])
labels_tensor_array.shape

loss_array = []
for i in range(labels_tensor_array.shape[0]):
    for j in range(labels_tensor_array.shape[1]):
        loss_array.append(-np.log(preds_array[labels_tensor_array[i][j]])[i][j])

mean_loss = np.array(loss_array).mean()
print(mean_loss)
labels_tensor.unique()
