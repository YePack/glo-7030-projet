import torch.nn.functional as F
from src.unet.unet_utils import *
from torch.nn import Linear, Softmax


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.lin1 = Linear(512*16*28, 512)
        self.lin2 = Linear(512, 128)
        self.lin3 = Linear(128, 9)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        x5_flatten = x5.view(x5.size()[0], -1)
        x_fully_proportion = self.lin1(x5_flatten)
        x_fully_proportion = self.lin2(x_fully_proportion)
        x_fully_proportion = self.lin3(x_fully_proportion)
        x_fully_proportion_out = self.softmax(x_fully_proportion)

        return x, x_fully_proportion_out
