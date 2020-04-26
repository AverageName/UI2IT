import sys
sys.path.append('..')
import torch
import torch.nn as nn
from utils.utils import get_activation, get_norm_module


class UpsampleConvBlock(nn.Module):
    
    def __init__(self, in_channels, kernel_size, activation, norm_type, pad_type):
        super(UpsampleConvBlock, self).__init__()
        
        if pad_type == "reflection":
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,
                              kernel_size=kernel_size)
        elif pad_type == "zeros":
            self.pad = None
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,
                              kernel_size=kernel_size, padding=kernel_size//2)
        
        self.act = get_activation(activation)
        self.upsample = nn.Upsample(scale_factor=2)
        self.norm = get_norm_module(norm_type)(in_channels//2)
    
    def forward(self, inputs):
        out = self.upsample(inputs)
        if self.pad is not None:
            out = self.pad(out)
        out = self.conv(out)
        out = self.norm(out)
        out = self.act(out)
        return out 