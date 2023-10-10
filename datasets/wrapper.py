from typing import Literal
import os

import torch
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms as T

class DatasetWrapper:
    def __init__(
        self,
        dataset: str,
        data_root: str = ".",
        seed: int = 2023
    ):
        super(DatasetWrapper, self).__init__()
        data = {k.lower(): v for k, v in datasets.__dict__.items()}[dataset]
        
        match dataset:
            case "mnist":
                norm_stats = ((0.1307,), (0.3081,))
            
            case "cifar10":
                norm_stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            
            case "cifar100":
                norm_stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            
            case "stl10":
                norm_stats = ((0.4467, 0.4398, 0.4066), (0.2603, 0.2565, 0.2712))
            
            case "imagenet":
                norm_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                
            case _:
                raise NotImplementedError

        if dataset in ("mnist", "cifar10", "cifar100"):
            image_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(*norm_stats)
            ])
            train_data = data(root = data_root, train = True, transform = image_transform, download = True)
            train_data_len = len(train_data)
            finetune_data_len = int(train_data_len * 0.1)
            train_data_len = train_data_len - finetune_data_len
            
            self.train_data, self.finetune_data = random_split(train_data, [train_data_len, finetune_data_len], generator=torch.Generator().manual_seed(seed))
            self.test_data = data(root = data_root, train = False, transform = image_transform, download = True)
            
        elif dataset == "imagenet":
            train_image_transform = T.Compose([
                T.RandomResizedCrop(224),
                T.ToTensor(),
                # T.Normalize(*norm_stats),
            ])
            
            test_image_transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                # T.Normalize(*norm_stats)
            ])
            train_data = data(root = os.path.join(data_root, "train"), split = 'train', transform = train_image_transform)
            train_data_len = len(train_data)
            finetune_data_len = int(train_data_len * 0.1)
            train_data_len = train_data_len - finetune_data_len
            
            self.train_data, self.finetune_data = random_split(train_data, [train_data_len, finetune_data_len], generator=torch.Generator().manual_seed(seed))
            self.test_data = data(root = os.path.join(data_root, "val"), split = 'val', transform = test_image_transform)
            
           
        elif dataset == "stl10":
            image_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(*norm_stats)
            ])
            
            # Use unlabeled split as train, train split as fine-tuning/linear probe set
            self.train_data = data(root = data_root, split = 'unlabeled', transform = image_transform, download = True)
            self.finetune_data = data(root = data_root, split = 'train', transform = image_transform, download = True)
            self.test_data = data(root = data_root, split = 'test', transform = image_transform, download = True)

        else:
            raise NotImplementedError
    
    def __call__(self, split: Literal['train', 'finetune', 'test']):
        return getattr(self, f"{split}_data")