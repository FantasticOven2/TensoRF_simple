import torch, re
import numpy as np
from torch import searchsorted
from kornia import create_meshgrid

def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: see original github repo
    Inputs: 
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], 
                                torch.ones_like(i)], -1) # (H, W, 3)
    return direction