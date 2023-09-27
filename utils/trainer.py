from typing import List, Callable, Any, Literal

from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim import Adam, SGD, RMSprop

from models import ResNet, EfficientNet, MobileNet
from datasets import ImageNet

class Trainer:
    def __init__(
        self, 
        backbone: str,
        pretrained: bool = False,
        
        from_ckpt: bool = False,
        ckpt_path: str = None,
        
        transforms: List[Callable[[Any], Any]] = None,
        target_transforms: List[Callable[[Any], Any]] = None, 
        
        dataset_name: Literal["imagenet", "cifar10", "cifar100", "stl10"] = "imagenet",
    ):
        pass
    
    def build_dataloader(self, dataset: Dataset, **kwargs):
        return DataLoader(dataset, **kwargs)
    
    
    
