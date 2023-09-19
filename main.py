import pytorch_lightning as pl
from model import ImageRegression
from data_utils import get_transform, get_loaders
from train_utils import get_checkpoint_callback, get_logger
from datasets import load_dataset
from plot_utils import plot_truth_vs_prediction

transform = get_transform()
raw_datasets = load_dataset("Niche-Squad/mock-dots","regression-one-class", download_mode="force_redownload")
train_loader, val_loader, test_loader = get_loaders(raw_datasets,transform,32)

model = ImageRegression()
checkpoint_callback = get_checkpoint_callback()
logger = get_logger()

trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=100, logger=logger)
trainer.fit(model, train_loader, val_loader)

save_path_validation = 'D:/Files/Plot/validation.png'
plot_truth_vs_prediction(model, val_loader,save_path_validation)

best_model_path = trainer.checkpoint_callback.best_model_path
trainer.test(dataloaders=test_loader,ckpt_path=best_model_path) 
save_path_test = 'D:/Files/Plot/test.png'
plot_truth_vs_prediction(model, test_loader,save_path_test)