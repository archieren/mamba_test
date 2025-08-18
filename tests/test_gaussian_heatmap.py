import numpy as np

import torch
import matplotlib.pyplot as plt

def test_maybe_numpy_indices():
    # prepare some coordinates
    x, y, z = np.indices((8, 8, 8))

    # draw cuboids in the top left and bottom right corners, and a link between
    # them
    cube1 = (x < 3) & (y < 3) & (z < 3)
    cube2 = (x >= 5) & (y >= 5) & (z >= 5)
    link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

    # combine the objects into a single boolean array
    voxelarray = cube1 | cube2 | link

    # set the colors of each object
    colors = np.empty(voxelarray.shape, dtype=object)
    colors[link] = 'red'
    colors[cube1] = 'blue'
    colors[cube2] = 'green'

    # and plot everything
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(voxelarray, facecolors=colors, edgecolor='k')
    plt.show()

def gaussian_heatmap_by_numpy(height, width, depth, center, std_dev=4):
    x_axis = np.arange(width).astype(np.float32) - center[0]
    y_axis = np.arange(height).astype(np.float32) - center[1]
    z_axis = np.arange(depth).astype(np.float32) - center[1]
    x, y, z = np.meshgrid(x_axis, y_axis,z_axis)
    
    res =   np.exp(-((x ** 2 + y ** 2 + z ** 2) / (2 * std_dev ** 2)))  
    print(res.shape)
    return res 
        
def gaussian_heatmap(height, width, depth, center, std_dev=4):
    """
    Args:
    - height (int): Height of the heatmap.
    - width (int): Width of the heatmap.
    - center (tuple): The (x, y) coordinates of the Gaussian peak.
    - std_dev (int, optional): Standard deviation of the Gaussian.

    """
    x_axis = torch.arange(width).float() - center[0]
    y_axis = torch.arange(height).float() - center[1]
    z_axis = torch.arange(depth).float() - center[1]
    x, y, z = torch.meshgrid(x_axis, y_axis,z_axis)
    print(x[0])
    
    return  torch.exp(-((x ** 2 + y ** 2 + z ** 2) / (2 * std_dev ** 2)))

def test_gaussian_heatmap():
    height, width, depth = 20, 20, 20
    center = (10, 10, 10)

    heatmap = gaussian_heatmap_by_numpy(height, width, depth, center)
    # heatmap = gaussian_heatmap(height, width, depth, center)   
    plt.imshow(heatmap[0], cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()
    
#test_maybe_numpy_indices() 
test_gaussian_heatmap()