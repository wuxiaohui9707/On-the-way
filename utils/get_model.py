from . import models
import torch
from pytorch_lightning.callbacks import ModelCheckpoint


def get_model(model_name, checkpoint_path=None):
    """return the model class"""
    model_class = None
    if "18" in model_name:
        model_class = models.ImageRegression_Resnet18()
    elif "34" in model_name:
        model_class = models.ImageRegression_Resnet34()
    elif "50" in model_name:
        model_class = models.ImageRegression_Resnet50()
    elif "101" in model_name:
        model_class = models.ImageRegression_Resnet101()
    elif "152" in model_name:
        model_class = models.ImageRegression_Resnet152()
    else:
        raise ValueError("Unsupported model_name: {}".format(model_name))
    
    if checkpoint_path is not None:
        model_class = model_class.load_from_checkpoint(best_model_path)
        model_class.load_state_dict(checkpoint['model_state_dict']) 
    return model_class