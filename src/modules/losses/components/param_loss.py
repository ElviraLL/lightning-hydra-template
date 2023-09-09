import torch
import torch.nn as nn
from utils.params import ParamsPack
param_pack = ParamsPack()
import math


class ParamLoss(nn.Module):
    """Input and target are all 62-d param"""
    def __init__(self):
        super(ParamLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, input, target, mode='normal'):
        if mode == 'normal':
            loss = self.criterion(input[:,:12], target[:,:12]).mean(1) + self.criterion(input[:,12:], target[:,12:]).mean(1)
            return torch.sqrt(loss)
        elif mode == 'only_3dmm':
            loss = self.criterion(input[:,:50], target[:,12:62]).mean(1)
            return torch.sqrt(loss)
        # return torch.sqrt(loss.mean(1))


if __name__ == "__main__":
    pass