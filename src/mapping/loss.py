import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceCoeff(nn.Module):
    """Dice coeff for batch examples"""

    def forward(self, input_label, h, plan_2d):
        input_prime = self.warpPespective(input_label, h, (510, 256))


    @staticmethod
    def warpPerspective(img, M, dsize):
        mtr = img
        C, R = dsize
        dst = np.full((C, R), 9.)
        for i in range(C):
            for j in range(R):
                res = np.dot(M, [j, i, 1])
                i2, j2, _ = (res / res[2] + 0.5).astype(int)
                if i2 >= 0 and i2 < R:
                    if j2 >= 0 and j2 < C:
                        dst[j2, i2] = mtr[i, j]
        return dst
    @staticmethod
    def losses(input_prime, plan_2d):
        # first mean image second mean in plan
        field_in_in = 0
        field_in_out = 0
        field_out_in = 0
        field_out_out = 0
        line = 0
        circle = 0

        #Je sais pas ce que je voulais faire ici.
        #for i in range(input_prime[0]):
            #for j in range(input_prime[1]):
                #if input_prime:

