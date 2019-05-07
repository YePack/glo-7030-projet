import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn

from src.net_parameters import p_number_of_classes


model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


ori_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])

adapt_state_dict = OrderedDict()
adapt_state_dict["features.0.weight"] = ori_state_dict["features.0.weight"]
adapt_state_dict["features.0.bias"] = ori_state_dict["features.0.bias"]
adapt_state_dict["features.1.weight"] = ori_state_dict["features.1.weight"]
adapt_state_dict["features.1.bias"] = ori_state_dict["features.1.bias"]
adapt_state_dict["features.1.running_mean"] = ori_state_dict["features.1.running_mean"]
adapt_state_dict["features.1.running_var"] = ori_state_dict["features.1.running_var"]
adapt_state_dict["features.3.weight"] = ori_state_dict["features.3.weight"]
adapt_state_dict["features.3.bias"] = ori_state_dict["features.3.bias"]
adapt_state_dict["features.4.weight"] = ori_state_dict["features.4.weight"]
adapt_state_dict["features.4.bias"] = ori_state_dict["features.4.bias"]
adapt_state_dict["features.4.running_mean"] = ori_state_dict["features.4.running_mean"]
adapt_state_dict["features.4.running_var"] = ori_state_dict["features.4.running_var"]
adapt_state_dict["features.6.weight"] = ori_state_dict["features.7.weight"]
adapt_state_dict["features.6.bias"] = ori_state_dict["features.7.bias"]
adapt_state_dict["features.7.weight"] = ori_state_dict["features.8.weight"]
adapt_state_dict["features.7.bias"] = ori_state_dict["features.8.bias"]
adapt_state_dict["features.7.running_mean"] = ori_state_dict["features.8.running_mean"]
adapt_state_dict["features.7.running_var"] = ori_state_dict["features.8.running_var"]
adapt_state_dict["features.9.weight"] = ori_state_dict["features.10.weight"]
adapt_state_dict["features.9.bias"] = ori_state_dict["features.10.bias"]
adapt_state_dict["features.10.weight"] = ori_state_dict["features.11.weight"]
adapt_state_dict["features.10.bias"] = ori_state_dict["features.11.bias"]
adapt_state_dict["features.10.running_mean"] = ori_state_dict["features.11.running_mean"]
adapt_state_dict["features.10.running_var"] = ori_state_dict["features.11.running_var"]
adapt_state_dict["features.12.weight"] = ori_state_dict["features.14.weight"]
adapt_state_dict["features.12.bias"] = ori_state_dict["features.14.bias"]
adapt_state_dict["features.13.weight"] = ori_state_dict["features.15.weight"]
adapt_state_dict["features.13.bias"] = ori_state_dict["features.15.bias"]
adapt_state_dict["features.13.running_mean"] = ori_state_dict["features.15.running_mean"]
adapt_state_dict["features.13.running_var"] = ori_state_dict["features.15.running_var"]
adapt_state_dict["features.15.weight"] = ori_state_dict["features.17.weight"]
adapt_state_dict["features.15.bias"] = ori_state_dict["features.17.bias"]
adapt_state_dict["features.16.weight"] = ori_state_dict["features.18.weight"]
adapt_state_dict["features.16.bias"] = ori_state_dict["features.18.bias"]
adapt_state_dict["features.16.running_mean"] = ori_state_dict["features.18.running_mean"]
adapt_state_dict["features.16.running_var"] = ori_state_dict["features.18.running_var"]
adapt_state_dict["features.18.weight"] = ori_state_dict["features.20.weight"]
adapt_state_dict["features.18.bias"] = ori_state_dict["features.20.bias"]
adapt_state_dict["features.19.weight"] = ori_state_dict["features.21.weight"]
adapt_state_dict["features.19.bias"] = ori_state_dict["features.21.bias"]
adapt_state_dict["features.19.running_mean"] = ori_state_dict["features.21.running_mean"]
adapt_state_dict["features.19.running_var"] = ori_state_dict["features.21.running_var"]
adapt_state_dict["conv_out.weight"] = nn.init.normal(torch.zeros((p_number_of_classes, 256, 1, 1)), 0)
adapt_state_dict["conv_out.bias"] = torch.zeros(p_number_of_classes)
