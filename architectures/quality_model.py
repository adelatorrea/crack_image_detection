import torch
import torch.nn as nn
import torchvision.models as models



class quality_classif(nn.Module):
    def __init__(self, num_classes=1):
        super(quality_classif, self).__init__()
        
        # Load pretrained ResNet
        self.resnet = models.resnet18(pretrained=False)
        
        in_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)
        
        