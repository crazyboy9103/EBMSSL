from typing import Any, Callable

from torchvision.datasets import CIFAR10, CIFAR100

from .base import BaseDataset

class CIFAR(BaseDataset):
    def __init__(
        self,
        data_path: str, 
        transform: Callable[[Any], Any] = None, 
        target_transform: Callable[[Any], Any] = None,
        **kwargs
    ):
        super(CIFAR, self).__init__(data_path, transform, target_transform, **kwargs)
        
    def load_from_path(self, data_path: str, **kwargs):
        version = kwargs.pop("version", 10)
        if version == 10:
            return CIFAR10(data_path, **kwargs)
        
        elif version == 100:
            return CIFAR100(data_path, **kwargs)
        
        else:
            raise ValueError(f"version must be 10 or 100, but got {version}")
        
        
        
    
