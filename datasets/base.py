from typing import Any, Callable

import torch
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        transform: Callable[[Any], Any] = None, 
        target_transform: Callable[[Any], Any] = None,
    ):
        self.data = self.load_from_path(data_path)
        self.transform = transform
        self.target_transform = target_transform
    
    def load_from_path(self, data_path: str):
        # return torch.load(data_path)
        raise NotImplementedError
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return (image, label)
