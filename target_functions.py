import torch


# used in example 4.1
def f_max(x):
    return torch.max(x[:, 0], x[:, 1])
