import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchvision.models import efficientnet_b2
import torch.optim as optim
from typing import Tuple

class EfficientNetSTRIP(pl.LightningModule):
    """
    EfficientNetSTRIP: Class implementation for EfficientNet with STRIP for multi-class classification.

    Attributes:
        - efficientnet (efficientnet_b2): EfficientNet model with pre-trained weights for feature extraction.
        - softmax (nn.Softmax): Softmax activation for the final output layer.
        - criterion (nn.CrossEntropyLoss): Cross Entropy Loss for multi-class classification.

    Methods:
        - forward(self, x: torch.Tensor) -> torch.Tensor: Forward pass through the EfficientNetSTRIP network.
        - training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict: 
            Perform one training step and compute the loss.
        - validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict: 
            Perform one validation step and compute the loss and accuracy.
        - calculate_accuracy(self, targets: torch.Tensor, outputs: torch.Tensor) -> float: 
            Calculate the accuracy between predicted and target values.
        - configure_optimizers(self) -> Tuple[list, list]: Configure the optimizer and learning rate scheduler.

    Example:
        model = EfficientNetSTRIP(cfg)
    """

    def __init__(self, cfg: dict):
        super(EfficientNetSTRIP, self).__init__()
        self.efficientnet = efficientnet_b2(num_classes=2)
        self.save_hyperparameters()
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the EfficientNetSTRIP network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor representing the class probabilities after softmax.
        """
        return self.softmax(self.efficientnet(x))

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Perform one training step and compute the loss.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of training data.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: Dictionary containing training loss, labels, and outputs.
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels).item() / len(labels)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True, on_step=True)
        return {"loss": loss, "labels": labels, "outputs": outputs}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        """
        Perform one validation step and compute the loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Batch of validation data.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: Dictionary containing validation loss, labels, and outputs.
        """
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels).item() / len(labels)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, on_step=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=True)
        return {"val_loss": loss, "labels": labels, "outputs": outputs}

    def calculate_accuracy(self, targets: torch.Tensor, outputs: torch.Tensor) -> float:
        """
        Calculate the accuracy between predicted and target values.

        Args:
            targets (torch.Tensor): Target tensor.
            outputs (torch.Tensor): Output tensor.

        Returns:
            float: Accuracy value.
        """
        binary_predictions = (outputs > 0.5).float()
        correct_predictions = (binary_predictions == targets).float()
        accuracy = correct_predictions.mean().item()
        return accuracy

    def configure_optimizers(self) -> Tuple[list, list]:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Tuple[list, list]: Tuple containing the optimizer and learning rate scheduler lists.
        """
        optimizer = optim.Adam(self.parameters(), lr=0.00001)
        lr_scheduler = {'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=1e-7),
                        'monitor': 'train_loss'}

        return [optimizer], [lr_scheduler]
