import pytorch_lightning as pl
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch
from utils import data, get_model, plot, train
from datasets import load_dataset
from pytorch_lightning.callbacks import ModelCheckpoint

raw_datasets = load_dataset("Niche-Squad/mock-dots","regression-one-class", download_mode="force_redownload")
loss = []
for n_res in [18, 34, 50, 101, 152]:
    # load dataset
    transform = data.get_transform()
    train_loader, val_loader, test_loader = data.get_loaders(raw_datasets,transform,32)
    # configure model
    model_name = "ImageRegression_Resnet{}".format(n_res)
    model = get_model.get_model(model_name)
    # config trainer and fit
    
    trainer = pl.Trainer(callbacks=[train.get_checkpoint_callback()], max_epochs=10, logger=train.get_logger())
    trainer.fit(model, train_loader, val_loader)

    # vis: val
    save_path = "E:/Files/Plot"
    plot.truth_vs_prediction(model,val_loader,os.path.join(save_path, model_name + "val.png"),)

    # vis: test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = model.load_from_checkpoint(best_model_path)
    test_result = trainer.test(dataloaders=test_loader,ckpt_path=best_model_path) 
    print(test_result)
    plot.truth_vs_prediction(model,test_loader,os.path.join(save_path, model_name + "test.png"))
    loss.append(test_result[0]["test_loss_epoch"])

# plot loss
plt.plot(loss)  # how the loss changes over different models