import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
# import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class ConBRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(ConBRBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=dilation, bias=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return self.relu(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        reduced_channels = channels // reduction
        self.conv1 = nn.Conv1d(channels, reduced_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(reduced_channels, channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_se = F.adaptive_avg_pool1d(x, 1)
        x_se = self.relu(self.conv1(x_se))
        x_se = self.sigmoid(self.conv2(x_se))
        return x * x_se  # Element-wise multiplication

class ReBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ReBlock, self).__init__()
        self.cbr1 = ConBRBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation)
        self.cbr2 = ConBRBlock(out_channels, out_channels, kernel_size, stride=1, dilation=dilation)
        self.se_block = SEBlock(out_channels)
    
    def forward(self, x):
        x_res = self.cbr1(x)
        x_res = self.cbr2(x_res)
        x_res = self.se_block(x_res)
        return x + x_res  # Residual connection

class UNet1D(nn.Module):
    def __init__(self, input_channels=1, layer_channels=32, kernel_size=3, depth=3, input_length=72):
        super(UNet1D, self).__init__()
        self.input_channels = input_channels
        self.layer_channels = layer_channels
        self.kernel_size = kernel_size
        self.depth = depth
        self.input_length = input_length
        
        # Downsampling
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = self.down_layer(self.input_channels, self.layer_channels, self.kernel_size, stride=1, depth=2)
        self.enc2 = self.down_layer(self.layer_channels, self.layer_channels * 2, self.kernel_size, stride=1, depth=2)
        self.enc3 = self.down_layer(self.layer_channels * 2, self.layer_channels * 4, self.kernel_size, stride=1, depth=2)

        # Decoder
        self.up1 = ConBRBlock(self.layer_channels * 8, self.layer_channels * 2, self.kernel_size, stride=1, dilation=1)
        self.up2 = ConBRBlock(self.layer_channels * 4, self.layer_channels, self.kernel_size, stride=1, dilation=1)
        self.up3 = ConBRBlock(self.layer_channels * 2, self.layer_channels, self.kernel_size, stride=1, dilation=1)
        
        # Upsampling
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')

        # Final output layer
        self.final_conv = nn.Conv1d(self.layer_channels, 1, kernel_size=self.kernel_size, stride=1, padding=1)
    
    def down_layer(self, in_channels, out_channels, kernel_size, stride, depth):
        layers = [ConBRBlock(in_channels, out_channels, kernel_size, stride=stride, dilation=1)]
        for _ in range(depth - 1):
            layers.append(ReBlock(out_channels, out_channels, kernel_size, dilation=1))
        return nn.Sequential(*layers)
    
    def forward(self, x, t):
        # Concatenate time step information
        x = x.view(-1, 72)
        t = t.view(-1, 1).float() # Expanding to match input length
        x = torch.cat([x, t], dim=1)
        x = x.unsqueeze(1)

        # Pooling operations
        pool_x1 = self.pool1(x)
        pool_x2 = self.pool2(pool_x1)
        pool_x3 = self.pool3(pool_x2)

        # Encoder
        out_0 = self.enc1(x)
        out_1 = self.enc2(out_0)
        
        print(out_1.shape)
        print(pool_x1.shape)
        print(pool_x2.shape)
        print(pool_x3.shape)
        x = torch.cat([out_1, pool_x1], dim=1)
        out_2 = self.enc3(x)

        # Decoder
        up = self.upsample1(out_2)
        up = torch.cat([up, out_1], dim=1)
        up = self.up1(up)
        
        up = self.upsample2(up)
        up = torch.cat([up, out_0], dim=1)
        up = self.up2(up)
        
        up = self.upsample3(up)
        out = self.up3(up)

        # Final convolution
        out = self.final_conv(out)
        return out[..., :self.input_length].squeeze(1)