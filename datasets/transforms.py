import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class GriddedRandomMask:
    def __init__(self, mask_size, mask_ratio):
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio

    def __call__(self, x, batch_wise=False):
        batch_size, _, height, width = x.size()
        
        assert height % self.mask_size == 0 and width % self.mask_size == 0, "Height and width must be divisible by mask_size."
        
        num_grids_h = height // self.mask_size
        num_grids_w = width // self.mask_size
        num_grids = num_grids_h * num_grids_w

        num_grids_to_mask = int(num_grids * self.mask_ratio)
        
        
        mask = torch.ones((batch_size, 1, num_grids_h, num_grids_w), dtype=torch.float32, device=x.device)
        
        if num_grids_to_mask > 0:
            if batch_wise:
                mask_indices = torch.randperm(num_grids)[:num_grids_to_mask]
                mask_indices_h = mask_indices // num_grids_w
                mask_indices_w = mask_indices % num_grids_w
                mask[:, :, mask_indices_h, mask_indices_w] = 0.0
            
            else:
                for batch_idx in range(batch_size):
                    mask_indices = torch.randperm(num_grids)[:num_grids_to_mask]
                    mask_indices_h = mask_indices // num_grids_w
                    mask_indices_w = mask_indices % num_grids_w
                    mask[batch_idx, :, mask_indices_h, mask_indices_w] = 0.0

        mask = F.interpolate(mask, size=(height, width), mode='nearest')
        
        # Apply
        masked_image = x * mask

        return masked_image
    
class RandomMask:
    def __init__(self, mask_size, mask_ratio):
        self.mask_size = mask_size
        self.mask_ratio = mask_ratio

    def __call__(self, x, batch_wise=False):
        batch_size, _, height, width = x.size()
        num_masks = int(height * width * self.mask_ratio / (self.mask_size ** 2))

        mask = torch.ones((batch_size, 1, height, width), dtype=torch.float32, device=x.device)
        
        if num_masks > 0:
            if batch_wise:
                for _ in range(num_masks):
                    top_left_y = random.randint(0, height - self.mask_size)
                    top_left_x = random.randint(0, width - self.mask_size)
                    
                    mask[:, :, top_left_y:top_left_y+self.mask_size, top_left_x:top_left_x+self.mask_size] = 0.0
            else:
                for batch_idx in range(batch_size):
                    for _ in range(num_masks):
                        top_left_y = random.randint(0, height - self.mask_size)
                        top_left_x = random.randint(0, width - self.mask_size)
                        
                        mask[batch_idx, :, top_left_y:top_left_y+self.mask_size, top_left_x:top_left_x+self.mask_size] = 0.0

        masked_image = x * mask
        return masked_image
    
class SuperResolution:
    def __init__(self, scale_factor: int):
        self.scale_factor = scale_factor

    def __call__(self, x):
        _, _, height, width = x.size()
        assert height % self.scale_factor == 0 and width % self.scale_factor == 0, "Height and width must be divisible by scale_factor."
        down_sampled = F.interpolate(
            x, size = (height // self.scale_factor, width // self.scale_factor), mode='bicubic'
        )
        up_sampled = F.interpolate(
            down_sampled, size = (height, width), mode='nearest'
        )
        return up_sampled

class Noise:
    def __init__(self):
        pass
    
    def __call__(self, x, batch_wise=False):
        batch_size, _, height, width = x.size()
        if batch_wise:
            gamma = torch.rand((1,), device=x.device)
            epsilon = torch.rand((height, width), device=x.device)
            return torch.sqrt(gamma) * x + torch.sqrt(1-gamma) * epsilon
        
        gamma = torch.rand((batch_size, 1, 1, 1), device=x.device)
        epsilon = torch.rand((batch_size, 1, height, width), device=x.device)
        return torch.sqrt(gamma) * x + torch.sqrt(1-gamma) * epsilon