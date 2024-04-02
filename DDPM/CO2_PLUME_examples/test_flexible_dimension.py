'''
this file reproduce the training process from
https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
'''

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
import os
import glob
import sys

# Get the path of the current file
current_file_path = os.path.abspath(__file__)
# Get the directory of the current file
current_dir = os.path.dirname(current_file_path)
# Get the grandparent directory (two levels up)
grandparent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)

from utility import Plume2D_dataloader
from viz import show_generated_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Create the network
net = UNet2DModel(
    sample_size=(20,68),  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ), 
    up_block_types=(
        "AttnUpBlock2D", 
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",   # a regular ResNet upsampling block
      ),
) #<<<
net.to(device)

x = torch.rand(64, 1, 20, 68).to(device)
pred = net(x, 0).sample

# print(pred.shape)





