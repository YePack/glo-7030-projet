import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.mapping.utils import warpPerspective, crop_center
from src.semantic.utils.utils import readfile


class MatchingLoss(nn.Module):

    def __init__(self):
        super(MatchingLoss, self).__init__()
        rink_2d_label = readfile('data/rink/rink_label')
        self.rink_2d = torch.tensor(crop_center(rink_2d_label, 450, 256))
        self.cen = torch.tensor([[[256.], [128.], [1.]]]*2)

    def forward(self, input_label, h):
        H = h.reshape(h.shape[0], 3, 3)
        H_I = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return (((H - H_I)**2).sum(axis=(1, 2))).mean()
        # proj = warpPerspective(input_label, H)
        # return (-((proj == 1)*(self.rink_2d == 1)).sum(axis=(1, 2)) + (.001*(H.bmm(self.cen)[:, 1] - 128)**2).T).mean()
