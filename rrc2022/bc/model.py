from torch import nn
import torch

class BC(nn.Module):
    def __init__(self):
        super(BC, self).__init__()
        self.w1 = nn.Linear(97, 97)
        self.w2 = nn.Linear(97, 9)
        
    def forward(self,x):
        x = self.w1(x)
        x = self.w2(x)
        return x

