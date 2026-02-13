from .adapter import OrderPreservingModel
from .architectures.directmodel import DirectModel

def build_logit_model(num_classes, use_adapter=False):
    model = DirectModel()
    if use_adapter:
        model = OrderPreservingModel(model, num_classes)
    model.eval()
    return model