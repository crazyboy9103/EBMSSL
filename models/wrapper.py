import torch
from torch import nn
from torchvision import models

class ModelWrapper(nn.Module):
    def __init__(
        self,
        backbone: str,
        base_alpha: float = 0.1, 
        from_ckpt: str = "",
        pretrained: bool = False,
        num_classes: int = 10
    ):
        super(ModelWrapper, self).__init__()
        # TODO add more backbones
        assert backbone in dir(models), f"Backbone {backbone} not found in torchvision.models."
        self.backbone = models.__dict__[backbone](pretrained=pretrained)
        
        self.alpha = nn.Parameter(torch.tensor(base_alpha), requires_grad=True)
        
        if "efficientnet" in backbone or "mobilenet" in backbone or "resnext" in backbone:
            in_features = self.backbone.classifier[-1].in_features
            self.energy_head = nn.Linear(in_features, 1, bias=False)
            self.backbone.classifier = nn.Identity()
            
        elif "resnet" in backbone:
            in_features = self.backbone.fc.in_features
            self.energy_head = nn.Linear(in_features, 1, bias=False)
            self.backbone.fc = nn.Identity()
            
        else:
            raise NotImplementedError
        
        self.linear_head = nn.Linear(in_features, num_classes)

        if from_ckpt:
            state_dict = torch.load(from_ckpt)
            self.load_state_dict(state_dict)
            
    def forward(self, x, return_energy=False, return_logits=False):
        fx = self.backbone(x)
        if return_energy:
            return self.energy_head(fx)
        
        elif return_logits:
            return self.linear_head(fx)
        
        return fx