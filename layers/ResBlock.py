import sys
sys.path.append('..')
import torch
import torch.nn as nn
from utils.utils import get_norm_module
import torch.nn.functional as F


class ResBlock(nn.Module):
    
    def __init__(self, in_planes, kernel_size=3, padding=(1, "reflection"), norm="batch"):
        super(ResBlock, self).__init__()
        if padding[1] == "reflection":
            self.pad1 = nn.ReflectionPad2d(padding[0])
            self.pad2 = nn.ReflectionPad2d(padding[0])
            self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size)
            self.conv2 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size, padding=padding[0])
            self.conv2 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=kernel_size, padding=padding[0])
            self.pad1 = None
            self.pad2 = None
        self.norm1 = get_norm_module(norm)(in_planes)
        self.norm2 = get_norm_module(norm)(in_planes)
        
        
    def forward(self, inputs):
        out = inputs
        if self.pad1:
            out = self.pad1(out)
        out = self.conv1(out)
        out = F.relu(self.norm1(out))
        if self.pad2:
            out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + inputs