from torch import nn
import torch

class BC(nn.Module):
    def __init__(self):
        super(BC, self).__init__()
        self.layer_names = [f'w{i}' for i in range(1,22)]
        for ln in self.layer_names:
            setattr(self, ln, nn.Linear(97, 97))
        self.w22 = nn.Linear(97, 9)
        
    def forward(self,x):
        prev_x = x
        for i, layer_name in enumerate(self.layer_names):
            x = getattr(self, layer_name)(x)
            if i % 2 == 1:  # residual connection
                x += prev_x
                prev_x = x
        x = self.w22(x)
        return x

