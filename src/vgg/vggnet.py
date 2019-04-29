import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from src.vgg.weight_adapt import adapt_state_dict

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256], #'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}
model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=9, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()
        self.conv_out = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.conv_out(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v != 'M':
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(adapt_state_dict)
    #for i, param in enumerate(model.parameters()):
        #if i <= 39:
           # param.requires_grad = False
    return model


