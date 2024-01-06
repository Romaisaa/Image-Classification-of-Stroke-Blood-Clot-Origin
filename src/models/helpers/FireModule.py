import torch.nn as nn
import torch


class Fire(nn.Module):
    """
    Fire Module: A building block for the SqueezeNet architecture.

    Args:
        - inplanes (int): Number of input channels.
        - squeeze_planes (int): Number of channels after the squeeze convolution layer.
        - expand1x1_planes (int): Number of channels after the 1x1 expand convolution layer.
        - expand3x3_planes (int): Number of channels after the 3x3 expand convolution layer.

    Attributes:
        - inplanes (int): Number of input channels.
        - squeeze (nn.Conv2d): 1x1 convolution layer to reduce the number of input channels.
        - squeeze_activation (nn.ReLU): ReLU activation after the squeeze layer.
        - squeeze_normalization (nn.BatchNorm2d): Batch normalization after the squeeze activation.
        - expand1x1 (nn.Conv2d): 1x1 convolution layer for the first branch in the expand layer.
        - expand3x3 (nn.Conv2d): 3x3 convolution layer for the second branch in the expand layer with padding.
        - expand_activation (nn.ReLU): ReLU activation after the expand layer.
        - expand_normalization (nn.BatchNorm2d): Batch normalization after the expand activation.

    Methods:
        - forward(x: torch.Tensor) -> torch.Tensor: Forward pass through the Fire module.

    Example:
        fire_module = Fire(inplanes=96, squeeze_planes=16, expand1x1_planes=64, expand3x3_planes=64)
    """
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
        """
        Forward pass through the Fire module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the squeeze and expand operations.
        """
        x = self.squeeze(x)
        x = self.squeeze_normalization(x)
        x = self.squeeze_activation(x)
        x =  torch.cat([self.expand1x1(x),self.expand3x3(x)],1)
        x = self.expand_normalization(x)
        x = self.expand_activation(x)
        
        return x
