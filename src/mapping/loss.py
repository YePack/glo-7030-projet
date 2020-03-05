import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.mapping.utils import warpPerspective, crop_center
from src.semantic.utils.utils import readfile

class MatchingLoss(nn.Module):

    def __init__(self): 

        rink_2d_label = readfile('data/rink/rink_label')
        self.rink_2d = crop_center(rink_2d_label, 450, 256)
        self.cen = np.array([256,128,1])


    def forward(self, input_label, h):
        H = h.reshape(h.shape[0],3,3)
        proj = warpPerspective(input_label, H)
        -np.sum((proj == 1)*(self.rink_2d == 1)) + .001*(H.dot(self.cen)[1] - 128)**2