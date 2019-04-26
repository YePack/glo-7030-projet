import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCoeff(nn.Module):
    """Dice coeff for batch examples"""

    def forward(self, y_pred, y_true):
        y_true_one_hot = self._one_hot(y_true)
        y_true_one_hot.cuda()
        y_pred_softmax = F.softmax(y_pred, dim=1)
        y_pred_softmax.cuda()
        epsilon = 0.000001
        axes = tuple(range(2, len(y_pred_softmax.shape)))
        numerator = torch.FloatTensor(2).cuda() * (y_pred_softmax * y_true_one_hot).sum(dim=axes)
        denominator = y_pred_softmax.sum(dim=axes) + y_true_one_hot.sum(dim=axes)
        return torch.FloatTensor(1).cuda() - (numerator / (denominator + epsilon)).mean()

    @staticmethod
    def _one_hot(input, nb_class=9):
        label_1_hot = torch.FloatTensor(1, nb_class, input.shape[1], input.shape[2])
        for i in range(nb_class):
            label_1_hot[0][i] = input[0] == i
        return label_1_hot
