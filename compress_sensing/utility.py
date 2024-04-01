import numpy as np

def get_penetrate_index(senders, receivers):
    '''
    solve the problem using mirroring method, mirror is at the bottom:
    we can have multiple senderes and receivers
    '''
    index_collect = []
    for s in senders:
        for r in receivers:
            # get the line equation
            x1, y1 = s
            x2, y2 = r
            assert y1 == y2, 'senders and receivers should be at the same depth'
            y2 = -y2 # mirror
            
            # get int(y1) points between sender and receiver
            x = np.linspace(x1, x2, int(y1)*2+1)
            y = np.linspace(y1, y2, int(y1)*2+1)

            # round the points to integer
            x = np.floor(x).astype(int)
            y = np.floor(abs(y)).astype(int)

            # transfer x,y to  a set of points
            indexes = list(zip(x, y))
            # remove index if it is out of the reservoir
            indexes = [i for i in indexes if  i[1] < int(y1)]

            # get the index of points that penetrate the reservoir
            index_collect += indexes

    return list(set(index_collect)) # remove duplicates

## index to mask
def index_to_mask(indexes, shape):
    '''
    indexes: list of index
    shape: shape of the mask
    '''
    mask = np.zeros(shape)
    for i,j in indexes:
        mask[i][j] = 1
    return mask


