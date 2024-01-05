import pytorch_lightning as pl
import torch.nn as nn
import torch
from torchvision.models import efficientnet_b2
import torch.optim as optim



class EfficientNetSTRIP(pl.LightningModule):
    def __init__(self, cfg: dict):
        super(EfficientNetSTRIP, self).__init__()        
        self.efficientnet = efficientnet_b2(num_classes=1)
        self.save_hyperparameters()
        self.sigmoid = nn.Sigmoid()
        class_weights = torch.tensor([3])  
        self.criterion = nn.BCEWithLogitsLoss(class_weights)
    
    def forward(self, x):
        return self.sigmoid(self.efficientnet(x))
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.view(-1, 1).float())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True)
        train_acc = self.calculate_accuracy(labels.view(-1, 1).float(), outputs)
        self.log("train_acc", train_acc, on_epoch=True, prog_bar=True, on_step=True)
        return {"loss": loss, "labels": labels, "outputs": outputs}


    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.view(-1, 1).float())
        val_acc = self.calculate_accuracy(labels.view(-1, 1).float(),outputs)
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True, on_step=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=True)
        return {"val_loss": loss, "labels": labels, "outputs": outputs}


    def calculate_accuracy(self, targets, outputs):
        binary_predictions = (outputs > 0.5).float()
        correct_predictions = (binary_predictions == targets).float()        
        accuracy = correct_predictions.mean().item()
        return accuracy

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.00001)
        lr_scheduler = {'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3,min_lr=1e-7),
                'monitor': 'train_loss'}

        return [optimizer], [lr_scheduler]