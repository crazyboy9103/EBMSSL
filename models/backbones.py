from torchvision import models
from torch import nn

class ResNet(nn.Module):
    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super(ResNet, self).__init__()
        self.model = models.resnet.__dict__[model_name](*args, **kwargs)
        self.model.fc = nn.Identity()
        
    def forward(self, x):
        return self.model(x)

class EfficientNet(nn.Module):
    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super(EfficientNet, self).__init__()
        self.model = models.efficientnet.__dict__[model_name](*args, **kwargs)
        self.model.classifier = nn.Identity()
    
    def forward(self, x):
        return self.model(x)
        
class MobileNet(nn.Module):
    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super(MobileNet, self).__init__()
        self.model = models.mobilenet.__dict__[model_name](*args, **kwargs)
        self.model.classifier = nn.Identity()
    
    def forward(self, x):
        return self.model(x)