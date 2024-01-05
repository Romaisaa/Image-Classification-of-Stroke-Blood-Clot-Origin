import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

class ResNet50Model(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-7):
        super(ResNet50Model, self).__init__()

        # Load the pretrained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Modify the last fully connected layer to match the number of classes
        out_features = self.resnet50.fc.out_features
        # Additional layers
        self.dense1 = nn.Linear(out_features, 256)  # Adjust the output size
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(256, num_classes)  # Fix the number of output units
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        # Learning rate
        self.learning_rate = learning_rate

    def forward(self, x):
        # Feed data forward through the network
        x = self.resnet50(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

    def configure_optimizers(self):
        # Use Adam optimizer and reduce learning rate on plateau to avoid overshooting
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        # Calculate loss
        loss = self.criterion(outputs, y)
        _, predictions = torch.max(outputs, 1)
        # Calculate accuracy
        acc = torch.sum(predictions == y).item() / y.size(0)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs,y)
        _, predictions = torch.max(outputs, 1)
        acc = torch.sum(predictions == y).item() / y.size(0)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss