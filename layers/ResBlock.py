import sys
sys.path.append('..')
import torch
import torch.nn as nn
from utils.utils import get_norm_module


class ResBlock(nn.Module):
    
    def __init__(self, in_planes, norm="batch"):
        super(ResBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.pad2 = nn.ReflectionPad2d(1)
        self.norm1 = get_norm_module(norm)(in_planes)
        self.norm2 = get_norm_module(norm)(in_planes)
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3)
        
    def forward(self, inputs):
        out = self.conv1(self.pad1(inputs))
        out = F.relu(self.norm1(out))
        out = self.conv2(self.pad2(out))
        out = self.norm2(out)
        return out + inputs