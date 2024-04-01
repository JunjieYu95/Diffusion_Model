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
            masked_image = original_image*mask[0].cpu()
            collected_images.append(masked_image.cpu())
        for i in range(ncols):
            collected_images.append(repaint_images[g*ncols+i].permute(2,0,1))
    # for i in range(repaint_images.shape[0]):
    #     masked_image = original_images[i]*mask[0]
    #     collected_images.append(original_images[i].cpu())
    #     collected_images.append(masked_image.cpu())
    #     collected_images.append(repaint_images[i].permute(2,0,1))

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
    fig.savefig(save_fig_name, bbox_inches='tight', dpi = 300)

def show_generated_samples(sample_batches, ncols=4):
    '''
    the sample batches are directly from the diffuser pipeline,
    it should has size [batch, height, width, channel] and store in numpy array
    '''
    batch_size = sample_batches.shape[0]
    nrows = batch_size // ncols

    fig, ax = plt.subplots(nrows, ncols, figsize=(18,6))
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].imshow(sample_batches[i*ncols+j].squeeze(), 
                           cmap='plasma',
                           vmin=0, vmax=1)
            ax[i,j].axis('off')
            # show colorbar
            # if j == ncols-1:
            #     plt.colorbar(ax[i,j].imshow(sample_batches[i*ncols+j].squeeze(), cmap='plasma'), ax=ax[i,j])
    # plt.show()
    return fig


                


    



