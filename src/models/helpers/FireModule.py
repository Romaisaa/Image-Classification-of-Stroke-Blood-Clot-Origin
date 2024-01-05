import torch.nn as nn
import torch


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.squeeze_normalization = nn.BatchNorm2d(num_features= squeeze_planes)  
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)
        
        self.expand_normalization = nn.BatchNorm2d(num_features=expand1x1_planes + expand3x3_planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        x = self.squeeze_normalization(x)
        x = self.squeeze_activation(x)
        x =  torch.cat([self.expand1x1(x),self.expand3x3(x)],1)
        x = self.expand_normalization(x)
        x = self.expand_activation(x)
        
        return x
