import sys
sys.path.append('..')
import torch
import torch.nn as nn
from utils.utils import get_norm_module
import torch.nn.functional as F


class ConvNormRelu(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=(1, "zeros"),
                 stride=1, norm="batch", leaky=True, conv_type="forward"):
        super(ConvNormRelu, self).__init__()
        if padding[1] == "zeros":
            self.pad = None
            if conv_type == "forward":
                self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding[0])
            elif conv_type == "transpose":
                self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding[0], output_padding=padding[0])
        elif padding[1] == "reflection":
            if conv_type == "forward":
                self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride)
                self.pad = nn.ReflectionPad2d(padding[0])
            elif conv_type == "transpose":
                self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride, padding=padding[0], output_padding=padding[0])
                self.pad = None
            
            
        self.leaky = leaky
        self.norm = get_norm_module(norm)(out_channels)
        
    def forward(self, inputs):
        out = inputs
        if self.pad is not None:
            out = self.pad(out)
        out = self.conv(out)
        out = self.norm(out)
        if self.leaky:
            return F.leaky_relu(out, negative_slope=0.2)
        else:
            return F.relu(out)