import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceCoeff(nn.Module):
    """Dice coeff for batch examples"""

    def forward(self, input_label, plan_2d):



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
