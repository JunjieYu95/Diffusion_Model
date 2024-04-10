import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch



def compare_repaint(repaint_images, original_images, mask, 
                    binary = False,
                    save_fig_name ='repaint_comparison.png'):
    '''
    repainted_images: tensor [batch, channel, height, width]
    original_images: tensor [batch, channel, height, width]
    mask: tensor [1,1,height, width]
    '''
    collected_images = []
    batch_size = repaint_images.shape[0]
    ncols = 8
    assert batch_size % ncols == 0, "batch size should be divisible by ncols"
    group = batch_size // ncols
    for g in range(group):
        for i in range(ncols):
            original_image = original_images[g*ncols+i].cpu()*0.5+0.5
            collected_images.append(original_image)
        for i in range(ncols):
            original_image = original_images[g*ncols+i].cpu()*0.5+0.5
            masked_image = original_image*mask[g*ncols+i].cpu()
            # print(masked_image.shape)
            # print('masked',masked_image.min(), masked_image.max())
            collected_images.append(masked_image)
            # collected_images.append(mask[0].cpu())
        for i in range(ncols):
            # print(repaint_images[g*ncols+i].shape)
            # print(repaint_images[g*ncols+i].min(), repaint_images[g*ncols+i].max())
            collected_images.append(repaint_images[g*ncols+i].permute(2,0,1)*0.5+0.5)
        # compare the repaint images and original images only at the masked region


    collected_images = torch.stack(collected_images)
    if binary:
        collected_images = collected_images > 0.5
    
    viz_grid = torchvision.utils.make_grid(collected_images, nrow=ncols,
                                           padding=2, 
                                           pad_value=1).clip(0,1)

    # apply colormap
    viz_grid = viz_grid.permute(1,2,0).numpy()
    fig = plt.figure(figsize=(20,20))
    plt.imshow(viz_grid[:,:,0], cmap = 'plasma')
    plt.axis('off')

    # save the figure
    if save_fig_name:
        fig.savefig(save_fig_name, bbox_inches='tight', dpi = 300)

def show_conditional_generation(conditional_images, generated_images, ncols=4, binary=False):
    '''
    conditional_images: range (-1,1), we set backgraound to 0 
                        -1, 1 represents two categories for binary examples
    generated_imgeas: range (-1,1)
    '''
    batch_size = conditional_images.shape[0]
    nrows = batch_size // ncols
    if binary:
        generated_images = generated_images > 0
        # to [-1, 1]
        generated_images = generated_images*2-1

    fig, ax = plt.subplots(nrows*2, ncols, figsize=(12,6))
    for i in range(nrows):
        for j in range(ncols):
            ax[i*2,j].imshow(conditional_images[i*ncols+j].squeeze(), 
                           cmap='plasma',
                           vmin=-1, vmax=1)

            ax[i*2+1,j].imshow(generated_images[i*ncols+j].squeeze(), 
                           cmap='plasma',
                           vmin=-1, vmax=1)

    # # adjust the space between subplots
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # add label on y axis of the first column
    for i in range(nrows):
        ax[i*2,0].set_ylabel('Condition')
        ax[i*2+1,0].set_ylabel('Generation')
    # remove all axis ticks without removing the labels
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    return fig


def compare_generated_samples(training_sampes, generated_samples, ncols=8, binary = False):
    '''
    the sample batches are directly from the diffuser pipeline,
    it should has size [batch, height, width, channel] and store in numpy array
    '''
    batch_size = generated_samples.shape[0]
    nrows = batch_size // ncols * 2

    if binary:
        generated_samples = generated_samples > 0.5

    # stack the samples together
    sample_batches = np.concatenate([training_sampes, generated_samples], axis=0)
    print(sample_batches.shape)
    fig, ax = plt.subplots(nrows, ncols, figsize=(10,10))
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].imshow(sample_batches[i*ncols+j].squeeze(), 
                           cmap='plasma',
                           vmin=0, vmax=1)
            ax[i,j].axis('off')



                


    



