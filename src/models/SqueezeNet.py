from src.models.helpers import FireModule as Fire
import pytorch_lightning as pl
import torch.nn as nn


class SqueezeNet(pl.LightningModule):
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

        


    def forward(self, x):
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


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss(reduction="none")(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).item() / len(y)
        self.log("train_loss",loss.mean(),on_epoch = True,prog_bar = True, on_step = True)
        self.log("train_acc", acc,on_epoch = True,prog_bar = True, on_step = True)
        return loss.mean()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss(reduction="none")(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).item() / len(y)
        self.log("val_loss",loss.mean(),on_epoch = True,prog_bar = True, on_step = True)
        self.log("val_acc",acc,on_epoch = True,prog_bar = True, on_step = True)
        return {"val_loss": loss.mean(),"val_acc": acc}


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        lr_scheduler = {'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3,min_lr=1e-6),
                'monitor': 'train_loss'}

        return [optimizer], [lr_scheduler]
