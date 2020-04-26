import sys
sys.path.append('..')
import torch
import torch.nn as nn
from utils.utils import AdaLIN, get_activation


class AdaLINResBlock(nn.Module):
    
    def __init__(self, in_channels, kernel_size, activation, pad_type):
        super(AdaLINResBlock, self).__init__()
        
        if pad_type == "reflection":
            self.pad1 = nn.ReflectionPad2d(kernel_size // 2)
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)
            self.pad2 = nn.ReflectionPad2d(kernel_size // 2)
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size)
        elif pad_type == "zeros":
            self.pad1 = None
            self.pad2 = None
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, padding=kernel_size//2)
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, padding=kernel_size//2)
            
        self.norm1 = AdaLIN(in_channels)
        self.norm2 = AdaLIN(in_channels)
        self.act = get_activation(activation)
    
    def forward(self, inputs, gamma, beta):
        out = inputs
        if self.pad1 is not None:
            out = self.pad1(out)
        out = self.conv1(out)
        out = self.act(self.norm1(out, gamma, beta))
        if self.pad2 is not None:
            out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return out + inputs