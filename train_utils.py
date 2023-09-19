from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def get_checkpoint_callback():
    return ModelCheckpoint(
        monitor='val_loss',
        dirpath="D:/Files/Checkpoint",
        mode='min',
        filename='best-model-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        save_last=False
    )

def get_logger():
    return TensorBoardLogger("tb_logs", name="Resnet50_Regression_lr_0.001_callbacks_T")
