from src.models.helpers import FireModule as Fire
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim


class SqueezeNet(pl.LightningModule):
     """
    SqueezeNet: Class implemnation for SqueezeNet
    Attributes:
        - conv0 (nn.Conv2d): First convolutional layer with 5x5 kernel, 96 output channels, and 2 stride.
        - bn0 (nn.BatchNorm2d): Batch normalization layer after conv0.
        - relu0 (nn.ReLU): Rectified Linear Unit activation after bn0.
        - maxpool0 (nn.MaxPool2d): Max pooling layer with 3x3 kernel, 2 stride, and 1 padding.
        - fire1-8 (Fire): Fire modules responsible for feature extraction.
        - drop (nn.Dropout): Dropout layer with a dropout rate of 0.5.
        - conv1 (nn.Conv2d): Convolutional layer with 1x1 kernel, reducing the number of channels to 2048.
        - bn1 (nn.BatchNorm2d): Batch normalization layer after conv1.
        - relu1 (nn.ReLU): Rectified Linear Unit activation after bn1.
        - avgpool (nn.AvgPool3d): 3D average pooling layer with a kernel size of (1024, 14, 14).
        - flatten (nn.Flatten): Flatten layer to convert the output to a 1D tensor.
        - softmax (nn.Softmax): Softmax activation for classification output.

    Methods:
        - forward(x): Forward pass through the network.
        - training_step(batch, batch_idx): Perform one training step and compute the loss.
        - validation_step(batch, batch_idx): Perform one validation step and compute the loss and accuracy.
        - configure_optimizers(): Configure the optimizer and learning rate scheduler.

    Example:
        model = SqueezeNet()
    """

    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.conv0 = nn.Conv2d(3, 96, kernel_size=5, stride=2)
        self.bn0 = nn.BatchNorm2d(96)
        self.relu0 = nn.ReLU(inplace=True)
        
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,)
        
        self.fire1 =  Fire(96, 16, 64, 64)
        self.fire2 =  Fire(128, 16, 64, 64)
        self.fire3 =   Fire(128, 32, 128, 128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2,  padding=1)
        
        self.fire4 =  Fire(256, 32, 128, 128)
        self.fire5 =  Fire(256, 48, 192, 192)
        self.fire6 =   Fire(384, 48, 192, 192)
        self.fire7 =   Fire(384, 64, 256, 256)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2,  padding=1)
        self.fire8 = Fire(512, 64, 512, 512)
        self.drop = nn.Dropout(0.5)
        
        self.conv1 = nn.Conv2d(1024,2048, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(2048)
        self.relu1 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool3d(kernel_size=(1024,14, 14))
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SqueezeNet network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor representing the class probabilities after softmax.
        """
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.maxpool1(x)
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.maxpool2(x)
        x = self.drop(x)
        x = self.fire8(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x) 
        x = self.avgpool(x)        
        x = self.flatten(x)
        x = self.softmax(x)

        
        return x


    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform one training step and compute the loss and acuuracy.

        Args:
            batch: Batch of training data.
            batch_idx: Index of the current batch.

        Returns:
            torch.Tensor: Training loss for the current step.
        """
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss(reduction="none")(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).item() / len(y)
        self.log("train_loss",loss.mean(),on_epoch = True,prog_bar = True, on_step = True)
        self.log("train_acc", acc,on_epoch = True,prog_bar = True, on_step = True)
        return loss.mean()
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Perform one validation step and compute the loss and accuracy.

        Args:
            batch: Batch of validation data.
            batch_idx: Index of the current batch.

        Returns:
            dict: Dictionary containing validation loss and accuracy.
        """
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss(reduction="none")(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).item() / len(y)
        self.log("val_loss",loss.mean(),on_epoch = True,prog_bar = True, on_step = True)
        self.log("val_acc",acc,on_epoch = True,prog_bar = True, on_step = True)
        return {"val_loss": loss.mean(),"val_acc": acc}


    def configure_optimizers(self) -> Tuple[list, list]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            list: List containing the optimizer.
            list: List containing the learning rate scheduler.
        """
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        lr_scheduler = {'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3,min_lr=1e-6),
                'monitor': 'train_loss'}

        return [optimizer], [lr_scheduler]
