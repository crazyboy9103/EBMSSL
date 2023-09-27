from torchvision.transforms import (
    Resize, 
    RandomCrop,
    
)
from torch import nn

class RandomMask(nn.Module):
    def __init__(self, )