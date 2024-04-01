import numpy as np
from matrix_completion import svt_solve
from utility import low_rank_approximation, random_mask, patched_mask
import matplotlib.pyplot as plt

def experiements(image, mask):
    # use original nuclear-norm based matrix completion algorithm
    recovered_image = svt_solve(image, mask)
    mae = (np.abs(image - recovered_image)*mask).sum()/mask.sum()
    return recovered_image, mae

def main(image, low_rank_transform = True, patch_size = (5, 5), rank = 10):
    # show comprehensive comparsion

    # create ratio from 0.1 to 0.9 with total num_ratios
    num_ratios = 9
    mask_ratio = np.linspace(0.1, 0.9, num_ratios)
    # mask_ratio = [i*0.1 for i in range(1, 2)]
    if low_rank_transform:
        image = low_rank_approximation(image, rank)

    res_recover = {'uniform':[], 'patched':[]}
    res_mask = {'uniform':[], 'patched':[]}
    res_mae = {'uniform':[], 'patched':[]}

    min_mae = 1e10
    max_mae = 0

    for ratio in mask_ratio:
        uniform_mask = random_mask(image, ratio)
        res_mask['uniform'].append(uniform_mask)
        recovered_image, mae = experiements(image, uniform_mask)
        min_mae = min(min_mae, mae)
        max_mae = max(max_mae, mae)
        res_recover['uniform'].append(recovered_image) 
        res_mae['uniform'].append(mae)

        patch_mask = patched_mask(image, ratio, patch_size=patch_size)
        res_mask['patched'].append(patch_mask)
        recovered_image, mae = experiements(image, patch_mask)
        min_mae = min(min_mae, mae)
        max_mae = max(max_mae, mae)
        res_recover['patched'].append(recovered_image)
        res_mae['patched'].append(mae)
    
    # plot the result
    fig, axes = plt.subplots(4, num_ratios+2)
    fig.set_size_inches((num_ratios+2)*2+2, 8)
    cmap1 = 'plasma'
    cmap2 = 'binary'
    # show original image
    axes[0,0].imshow(image, cmap = cmap1 )
    axes[1,0].imshow(image, cmap = cmap1 )
    axes[2,0].imshow(image, cmap = cmap1 )
    axes[3,0].imshow(image, cmap = cmap1 )
    # show masked image and recovered image
    for i in range(num_ratios):
        axes[0, i+1].imshow(image*res_mask['uniform'][i], cmap = cmap1)
        axes[0, i+1].imshow(res_mask['uniform'][i], alpha = 0.5, cmap = cmap2)
        axes[1, i+1].imshow(res_recover['uniform'][i], cmap = cmap1)
        axes[2, i+1].imshow(res_mask['patched'][i]*image, cmap = cmap1)
        axes[2, i+1].imshow(res_mask['patched'][i], alpha = 0.5, cmap = cmap2)
        axes[3, i+1].imshow(res_recover['patched'][i], cmap = cmap1)

    # show mae
    axes[1,num_ratios+1].plot(mask_ratio, res_mae['uniform'], label = 'uniform')
    axes[1,num_ratios+1].set_ylim([min_mae, max_mae])
    axes[3,num_ratios+1].plot(mask_ratio, res_mae['patched'], label = 'patched')
    axes[3,num_ratios+1].set_ylim([min_mae, max_mae])

    # decorate the figure
    # remove all the axis
    for i in range(4):
        for j in range(0,num_ratios+1):
            axes[i,j].axis('off')
    axes[0,num_ratios+1].axis('off')
    axes[2,num_ratios+1].axis('off')
    

    # add title at y axis for the first column
    # axes[0,0].set_ylabel('Uniform Mask')
    # axes[1,0].set_ylabel('Uniform Mask (Recovered)')
    # axes[2,0].set_ylabel('Patched Mask')
    # axes[3,0].set_ylabel('Patched Mask (Recovered)')

    # add title at x axis for the first row
    axes[0,0].set_title('Original Image')
    for i in range(num_ratios):
        # truncate the number to 2 decimal places
        axes[0, i+1].set_title('%.2f' % mask_ratio[i])
    axes[1, num_ratios+1].set_title('MAE')
    

    # save the figure
    plt.savefig('plume_result_fullrank_patch10.png')

if __name__ == '__main__':
    images = np.load('test_plume_data.npy')
    image = images[0]
    # image = np.load('pepper.npy')
    main(image,low_rank_transform = False,patch_size=(10,10))






    



