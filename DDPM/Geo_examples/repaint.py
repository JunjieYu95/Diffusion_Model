##
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel, DDPMPipeline
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
from utility import get_penetrate_mask, get_random_pixel_mask
from utility import Plume2D_dataloader, sis_dataloader, fluvial_dataloader

# define the model
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

# Load the model
model_path = '/scratch1/junjieyu/Diffusion_Model/DDPM/Geo_examples/monitor_channel/model_50.pt'
net.load_state_dict(torch.load(model_path))


from diffusers import RePaintPipeline, RePaintScheduler
generator = torch.Generator(device="cuda").manual_seed(0)

repaint_scheduler = RePaintScheduler.from_pretrained("google/ddpm-ema-celebahq-256")
repaint_pipe = RePaintPipeline(net, scheduler=repaint_scheduler)
repaint_pipe = repaint_pipe.to("cuda")

train_dataloader = fluvial_dataloader(batch_size = 16)
sample = next(iter(train_dataloader))
sample = sample.to(device)

# create mask (for plume dataset)
mask = get_random_pixel_mask(shape = (64,64), percentage = 0.01)
mask = mask.to(device)

repaint_images = repaint_pipe(
            image=sample,
            mask_image=mask,
            num_inference_steps=1000,
            eta=0.0,
            jump_length=10,
            jump_n_sample=5,
            generator=generator,
            output_type = 'numpy',
            ).images

from viz import compare_repaint
save_fig_name = 'test_repaint.png'
# to tensor
repaint_images = torch.tensor(repaint_images)
compare_repaint(repaint_images, sample, mask, 
                save_fig_name = save_fig_name)



