import sys
sys.path.append('..')
import torch
import torch.nn as nn
from utils.utils import get_activation

class ConvSpectralNormAct(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, activation):
        super(ConvSpectralNormAct, self).__init__()
        if padding[1] == "reflection":
            self.pad = nn.ReflectionPad2d(padding[0])
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride))
        elif padding[1] == "zeros":
            self.pad = None
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                 padding=padding[0], stride=stride))
        
        self.act = get_activation(activation)
        
    def forward(self, inputs):
        out = inputs
        if self.pad is not None:
            out = self.pad(out)
        out = self.conv(out)
        return self.act(out)