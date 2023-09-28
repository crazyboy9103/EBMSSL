from torch.nn import functional as F
from torch import nn

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, **kwargs) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes, **kwargs)
    
    def forward(self, x):
        return self.linear(x)
    