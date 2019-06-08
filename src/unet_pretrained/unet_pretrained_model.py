import torch.nn.functional as F
from src.unet_pretrained.unet_pretrained_utils import *
from src.unet_pretrained.unet_weight import adapt_state_dict


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 64)  # Les deux 64 ok
        self.down1 = downdouble_max(64, 128)  # les deux 128 ok
        self.down2 = downtriple(128, 256)  # les 3 256 ok
        self.down3 = downdouble(256, 256)
        #self.down3 = down(256, 512)
        #self.down4 = down(512, 512)
        #self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


def unet_pretrained(n_channels, n_classes):
    model = UNet2(n_channels, n_classes)
    for key in model.state_dict().keys():
        if key not in adapt_state_dict:
            adapt_state_dict[key] = model.state_dict()[key]
    model.load_state_dict(adapt_state_dict)
    return model
