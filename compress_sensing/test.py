from utility import get_penetrate_index
from utility import index_to_mask
import numpy as np
from matrix_completion import svt_solve, pmf_solve, nuclear_norm_solve

receivers_table = [] 
intervals = np.linspace(2,12,6)
for inv in intervals:
    receivers = []
    x = 0
    while x<68:
        receivers.append([int(x), 20])
        x += inv
    receivers_table.append(receivers)

senders = [[0,20],[67,20]]
for i in range(len(receivers_table)):
    receivers = receivers_table[i]
    grid = np.zeros((20,68))
    p_index = get_penetrate_index(senders, receivers)
    mask = index_to_mask(p_index, shape = (68,20))
    # reverse the mask, so all unavailable cell are 1, represents the mask
    mask_ratio = 1 - sum(mask.flatten())/mask.size
    # round the mask ratio to 2 decimal places
    ratio = round(mask_ratio, 2)

    plume_dataset = np.load('initial_explore/plume_dataset_vertical.npy')
    image = plume_dataset[15,-1,:,:]
    # show the image
    print(i)
    print(image.shape)
    # recovered_image = svt_solve(image, mask)
    recovered_image = nuclear_norm_solve(image, mask)

    # visualize the original image mask and recovered image
    import matplotlib.pyplot as plt
    cmap1 = 'plasma'
    cmap2 = 'binary'
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(image, cmap = cmap1)
    ax[1].imshow(mask*image, cmap = cmap1)
    ax[1].imshow(mask, cmap = cmap2, alpha = 0.5)
    ax[2].imshow(recovered_image, cmap = cmap1)

    # add title, origin, mask with ratio, recovered
    ax[0].set_title('Original Image')
    ax[1].set_title('Mask: ('+str(round(ratio*100,2))+'%)')
    ax[2].set_title('Recovered Image')
    # plt.show()
    # save the figure
    fig.savefig('recovered_image_vertical_Two_Sender'+str(i+1)+'.png')
