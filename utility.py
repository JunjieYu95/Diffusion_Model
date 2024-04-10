import torch
import torch.nn as nn
import numpy as np
import torchvision

def corrupt(x, amount):
  """Corrupt the input `x` by mixing it with noise according to `amount`"""
  noise = torch.rand_like(x)
  amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
  return x*(1-amount) + noise*amount

# mnist dataset for testing
def mnist_dataloader(batch_size=16, shuffle=True, num_workers=1):
    # Load the dataset using torchvision
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # remove labels
    dataset = [x[0] for x in dataset]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=shuffle, num_workers=num_workers)
    return dataloader

# createa a dataloader from geo_dataset
def Plume2D_dataloader(batch_size=16, shuffle=True, num_workers=1):
    file = '/scratch1/junjieyu/Diffusion_Model/geo_dataset/CO2_plume_vertical_2D_random.pt'
    # Load the dataset using torch.load
    dataset = torch.load(file)
    # add a channel dimension
    dataset = dataset.unsqueeze(1)
    # change the range to [-1,1]
    dataset = dataset * 2 - 1
    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=shuffle, num_workers=num_workers)
    return dataloader

def sis_dataloader(batch_size=16, shuffle=True, num_workers=1):
    file = '/scratch1/junjieyu/Diffusion_Model/geo_dataset/uncond-sis-train.pt'
    # Load the dataset using torch.load
    dataset = torch.load(file)
    # shift the range to [-1,1]
    dataset = dataset * 2 - 1
    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=shuffle, num_workers=num_workers)
    return dataloader

def fluvial_dataloader(batch_size=16, shuffle=True, num_workers=1):
    file = '/scratch1/junjieyu/Diffusion_Model/geo_dataset/fluvial-train.pt'
    # Load the dataset using torch.load
    dataset = torch.load(file)
    # shift the range to [-1,1]
    dataset = dataset * 2 - 1
    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                             shuffle=shuffle, num_workers=num_workers)
    return dataloader

def get_penetrate_mask(senders, receivers, shape = (68,68,20)):
    '''
    solve the problem using mirroring method, mirror is at the bottom:
    we can have multiple senderes and receivers
    the senders and receivers are located on the surface of the reservoir
    each sender and receiver is described by a tuple (x,y,z), where x,y,z are the coordinates
    '''
    index_collect = []
    for s in senders:
        for r in receivers:
            # get the line equation
            x1, y1, z1  = s
            x2, y2, z2 = r
            assert z1 == z2, 'senders and receivers should be at the same depth'
            z2 = -z2 # mirror

            # interpolate the path between the point (x1,y1,z1) and (x2,y2,z2)
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            x = np.linspace(x1, x2, int(distance)*2+1)
            y = np.linspace(y1, y2, int(distance)*2+1)
            z = np.linspace(z1, z2, int(distance)*2+1)

            # round the points to integer
            x = np.floor(x).astype(int)
            y = np.floor(y).astype(int)
            z = np.floor(abs(z)).astype(int)

            # transfer x,y,z to  a set of points
            indexes = list(zip(x, y, z))

            # remove index if it is out of the reservoir
            indexes = [i for i in indexes if  i[2] < z1]

            # get the index of points that penetrate the reservoir
            index_collect += indexes

    # remove duplicates
    index_collect = list(set(index_collect))

    # transform the index to mask
    mask = np.zeros(shape)
    for i,j,k in index_collect:
        mask[i][j][k] = 1

    # transfer the mask as a format that can be directly used by repaint pipeline
    mask = torch.tensor(mask.copy())

    # flip the maske bottom to top
    mask = mask.flip(2)
    
    # squuze and reshape
    mask = mask.unsqueeze(1).permute(0, 1, 3, 2)
    return mask

def get_random_pixel_mask(shape = (64,64), percentage = 0.5):
    mask = np.random.choice([0, 1], size=shape, p=[1-percentage, percentage])
    # transofrm the mask as a format that can be directly used by repaint pipeline
    mask = torch.tensor(mask.copy())
    # unsquuze
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

def show_3d_mask(mask):
    '''
    mask: 3d numpy array
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(mask, edgecolor='k')
    plt.show()

def patched_mask(shape, percentage, patch_size = (5,5)):
    image_size = shape
    mask = np.zeros(image_size, dtype=int)
    patch_rows, patch_cols = patch_size

    num_patches = int((image_size[0] / patch_rows) * (image_size[1] / patch_cols))
    num_selected_patches = int(num_patches * percentage)

    patch_indices = np.random.choice(num_patches, num_selected_patches, replace=False)

    for idx in patch_indices:
        i = (idx // (image_size[1] // patch_cols)) * patch_rows
        j = (idx % (image_size[1] // patch_cols)) * patch_cols
        mask[i:i+patch_rows, j:j+patch_cols] = 1

    mask = torch.tensor(mask.copy())
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask