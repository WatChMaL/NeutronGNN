from torch.optim import *

def select_optimizer(optimizer, model_params, **kwargs):
    if optimizer == "Adam":
        return Adam(model_params, **kwargs)
    elif optimizer == "SGD":
        return SGD(model_params, **kwargs)
    else:
        print("Optimizer {} not found".format(optimizer))
        return None
