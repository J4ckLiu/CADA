import torch

from .adapter import OrderPreservingModel
from .architectures.densenet import densenet121,densenet161
from .architectures.resnet import resnet101
from .architectures.resnext import resnext50



def build_model(model_name, num_classes=100, use_adapter=False):
    if model_name == "resnet101":
        model = resnet101()
        model.load_state_dict(torch.load('../model/model_weights/resnet101.pth'))
    elif model_name == "densenet161":
        model = densenet161()
        model.load_state_dict(torch.load('../model/model_weights/densenet161.pth'))
    elif model_name == "densenet121":
        model = densenet121()
        model.load_state_dict(torch.load('../model/model_weights/densenet121.pth'))
    elif model_name == "resnext50":
        model = resnext50()
        model.load_state_dict(torch.load('../model/model_weights/resnext50.pth'))
    else:
        raise ValueError(f"{model_name} is not supported")
    # If use_adapter is True, wrap the model with OrderPreservingModel
    if use_adapter:
        model = OrderPreservingModel(model, num_classes)
    model.eval()
    return model

