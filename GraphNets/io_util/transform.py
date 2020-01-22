import torch

def transform_reshape(new_shape):
    def __reshape(data):
        data.x = torch.reshape(data.x, new_shape)
        return data
    return __reshape

