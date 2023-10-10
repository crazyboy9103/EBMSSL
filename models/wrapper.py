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
        num_classes: int = 10, 
        group_norm: bool = True,
        groups: int = 32,
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
        
        if group_norm:
            self.swap_bn_to_gn(self.backbone, num_groups=groups)    
        
        
        self.linear_head = nn.Linear(in_features, num_classes)

        if from_ckpt:
            state_dict = torch.load(from_ckpt)
            self.load_state_dict(state_dict)
    
    def swap_bn_to_gn(self, module, num_groups=32):
        """
        Recursively swap all BatchNorm layers with GroupNorm layers in the module and its sub-modules.
        
        Parameters:
            module (torch.nn.Module): The module containing the BatchNorm layers to be replaced.
            num_groups (int): Number of groups to use for GroupNorm.
        """
        for name, child in list(module.named_children()):  # Convert to list to avoid dictionary size change during iteration
            if isinstance(child, nn.BatchNorm2d):
                # Create a new GroupNorm layer
                gn = nn.GroupNorm(num_groups, child.num_features, eps=child.eps, affine=child.affine)
                
                # Optionally, copy the parameters
                if child.affine:
                    gn.weight.data = child.weight.data.clone().detach()
                    gn.bias.data = child.bias.data.clone().detach()
                
                # Replace the child (BatchNorm layer) with the new layer (GroupNorm) in-place
                module._modules[name] = gn
                
            # Recur for child sub-modules
            self.swap_bn_to_gn(child, num_groups)
        
    def forward(self, x, return_energy=False, return_logits=False):
        fx = self.backbone(x)
        if return_energy:
            return self.energy_head(fx)
        
        elif return_logits:
            return self.linear_head(fx)
        
        return fx