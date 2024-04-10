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


train_dataloader = Plume2D_dataloader(batch_size = 16)


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

# setup training configuration
class TrainingConfig():
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
config = TrainingConfig()

# evalution during training
def evaluate(config, epoch, pipeline, losses):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
        output_type = 'numpy',
        num_inference_steps = 1000
    ).images

    # Show the images
    fig = show_generated_samples(images, ncols=4)

    # Save the images
    test_dir = os.path.join("monitor")
    os.makedirs(test_dir, exist_ok=True)
    
    fig.savefig(os.path.join(test_dir, f"epoch_{epoch}.png"))

    # losses figure
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    fig.savefig(os.path.join(test_dir, f"losses.png"))

# setup optimizer and scheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3) 
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

losses = []
# train the model
for epoch in range(config.num_epochs):
    for batch in train_dataloader:
        clean_images = batch
        clean_images = clean_images.to(device)
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
            dtype=torch.int64
        )
        # timesteps = torch.tensor([0]*bs)
        # timesteps = timesteps.to(device)
        # print('timesteps2',timesteps.shape)
        # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = net(noisy_images, timesteps, return_dict=False)[0]

        # l1 loss
        # loss = F.l1_loss(noise_pred, noise) #
        loss = F.mse_loss(noise_pred, noise) #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().item())
        
    print(f"Epoch {epoch}, Loss: {loss.item()}")
    if epoch % config.save_image_epochs == 0:
        pipeline = DDPMPipeline(unet=net, scheduler=noise_scheduler)
        evaluate(config, epoch, pipeline, losses)
        # evaluate2(net, noise_scheduler, config.eval_batch_size, epoch)

        # save the model
        model_dir = os.path.join("monitor")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(model_dir, f"model_{epoch}.pt"))



