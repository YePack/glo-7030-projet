import torch.nn.functional as F
from src.semantic.unet.unet_utils import *
from torch.nn import Linear


class HomographyNet(nn.Module):
    def __init__(self):
        super(HomographyNet, self).__init__()
        self.inc = inconv(9, 64)
        self.down1 = down(64, 128)
        #self.down2 = down(128, 128)
        #self.down3 = down(128, 128)
        self.lin1 = Linear(128 * 128 * 225, 1024)
        self.lin2 = Linear(1024, 9)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        #x = self.down2(x)
        #x = self.down3(x)
        x = x.view(x.size()[0], -1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x
