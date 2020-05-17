import sys
sys.path.append('..')
import torch
import torch.nn as nn
from utils.utils import get_norm_module, get_activation


class LinearNormAct(nn.Module):
    
    def __init__(self, in_channels, out_channels, norm='batch', activation="relu"):
        super(LinearNormAct, self).__init__()
        
        self.fc = nn.Linear(in_channels, out_channels)
        self.norm = get_norm_module(norm)(out_channels)
        self.activation = get_activation(activation)
    
    def forward(self, inputs):
        out = self.fc(inputs)
        out = self.norm(out)
        if self.activation is not None:
            return self.activation(out)
        else:
            return out