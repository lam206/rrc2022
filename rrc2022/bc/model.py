from torch import nn
import torch

class BC(nn.Module):
    def __init__(self):
        super(BC, self).__init__()
        self.w1 = nn.Linear(97, 97)
        self.w2 = nn.Linear(97, 97)
        self.w3 = nn.Linear(97, 97)
        self.w4 = nn.Linear(97, 97)
        self.w5 = nn.Linear(97, 97)
        self.w6 = nn.Linear(97, 97)
        self.w7 = nn.Linear(97, 97)
        self.w8 = nn.Linear(97, 97)
        self.w9 = nn.Linear(97, 97)
        self.w10 = nn.Linear(97, 97)
        self.w11 = nn.Linear(97, 97)
        self.w12 = nn.Linear(97, 97)
        self.w13 = nn.Linear(97, 97)
        self.w14 = nn.Linear(97, 97)
        self.w15 = nn.Linear(97, 97)
        self.w16 = nn.Linear(97, 97)
        self.w17 = nn.Linear(97, 97)
        self.w18 = nn.Linear(97, 97)
        self.w19 = nn.Linear(97, 97)
        self.w20 = nn.Linear(97, 97)
        self.w21 = nn.Linear(97, 9)
        
    def forward(self,x):
        layers = vars(self)
        prev_x = x
        for i, layer_name in enumerate(layers):
            x = layers[layer_name](x)
            if i % 2 == 1:  # residual connection
                x += prev_x
                prev_x = x
        return x

