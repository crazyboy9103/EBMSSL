from torchvision.transforms import (
    Resize, 
    RandomCrop,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

class RandomMask(nn.Module):
    def __init__(self, grid_size, probability):
        super(RandomMask, self).__init__()
        self.grid_size = grid_size
        self.probability = probability

    def forward(self, x):
        # x.sape = (batch_size, channels, height, width)
        batch_size, num_channels, height, width = x.size()
        
        num_grids_h = height // self.grid_size
        num_grids_w = width // self.grid_size
        num_grids = num_grids_h * num_grids_w

        num_grids_to_mask = int(num_grids * self.probability)
        
        mask = torch.ones((batch_size, 1, num_grids_h, num_grids_w), dtype=torch.float32, device=x.device)
        
        if num_grids_to_mask > 0:
            mask_indices = torch.randperm(num_grids)[:num_grids_to_mask]
            mask_indices_h = mask_indices // num_grids_w
            mask_indices_w = mask_indices % num_grids_w
            mask[:, :, mask_indices_h, mask_indices_w] = 0.0

        mask = F.interpolate(mask, size=(height, width), mode='nearest')
        
        # Apply
        masked_image = x * mask

        return masked_image