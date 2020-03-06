import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.mapping.utils import warpPerspective, crop_center
from src.semantic.utils.utils import readfile


class MatchingLoss(nn.Module):

    def __init__(self):
        super(MatchingLoss, self).__init__()

        self.xx = torch.tensor([[0],[0]],[[0],[0]],[[75],[375]],[[225],[0]],[[0],[0]],[[0],[0]],[[0],[0]],[[170],[280]])
        self.yy = torch.tensor([[0],[0]],[[0],[0]],[[70],[180]],[[128],[0]],[[0],[0]],[[0],[0]],[[0],[0]],[[70],[180]])

    def forward(self, input_label, h, device):

        projs = warpPerspectiveBatch(input_label, h, device)
        loss = 0
        for k in (3,4,8):
            for i in range(b):
                x = xx[k]
                y = yy[k]
                p = torch.where(projs[i]==k)
                
                if k == 4:
                    x = x[0]
                    y = y[0]
                
                dx = (p[0] - x)**2
                dy = (p[1] - y)**2

                if k == 3 | k == 8:
                    dx = torch.min(dx,dim=0).values
                    dy = torch.min(dy,dim=0).values
            
                loss = loss + torch.sum(dx+dy)/max(1,len(p[0]))
        return loss
