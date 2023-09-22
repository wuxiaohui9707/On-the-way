import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

###Resnet50###
class ImageRegression_Resnet50(pl.LightningModule):
    def __init__(self):
        super(ImageRegression_Resnet50, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x): # x: [batch_size, 3, 224, 224]
        x = x.float()  # 转换数据类型为float32
        return self.model(x).squeeze(-1)  # [batch_size, 1] -> [batch_size, ]

    def training_step(self, batch, batch_idx):
        images, labels = batch # images: [batch_size, 3, 224, 224], labels: [batch_size, 1]
        outputs = self(images)
        loss = F.mse_loss(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        # MSE loss
        loss = F.mse_loss(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # Accuracy
        rounded_outputs = torch.round(outputs)
        correct = (rounded_outputs == labels).sum().item()
        total = len(labels)
        accuracy = correct / total
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_acc": accuracy}

    def test_step(self,batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.mse_loss(outputs, labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        rounded_outputs = torch.round(outputs)
        correct = (rounded_outputs == labels).sum().item()
        total = len(labels)
        accuracy = correct / total
        self.log('test_acc', accuracy, on_epoch=True, prog_bar=True)
        return {"test_loss": loss, "test_acc": accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    