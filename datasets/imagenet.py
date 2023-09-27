from typing import Any, Callable

import torch

from .base import BaseDataset

class ImageNet(BaseDataset):
    def __init__(
        self,
        data_path: str, 
        transform: Callable[[Any], Any] = None, 
        target_transform: Callable[[Any], Any] = None,
    ):
        super(ImageNet, self).__init__(data_path, transform, target_transform)
        
    def load_from_path(self, data_path: str):
        return torch.load(data_path)
    
