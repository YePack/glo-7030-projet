import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.mapping.utils import warpPerspective, crop_center
from src.semantic.utils.utils import readfile

class DiceCoeff(nn.Module):
    """Dice coeff for batch examples"""

    def forward(self, input_label, h):
        input_prime = warpPerspective(input_label, h)
        rink_2d_label = readfile('data/rink/rink_label')
        rink_2d_label_crop = crop_center(rink_2d_label, 450, 256)


    @staticmethod
    def losses(input_prime, plan_2d):
        # first mean image second mean in plan
        field_in_in = 0
        field_in_out = 0
        field_out_in = 0
        field_out_out = 0
        line = 0
        circle = 0
        for i in range