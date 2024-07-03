import torch


# used in example 4.1
def f_max(x):
    return torch.max(x[:, 0], x[:, 1])


# for example 4.2

#ToDo: Load the Parameter alpha from example_4_2.yaml instead of hardcoding it as 0.7
def f_avar(x, tau, alpha):
    return tau + torch.mul(torch.tensor(1 / (1 - alpha)), torch.maximum(x[:, 0] + x[:, 1] - tau, torch.tensor(0.0)))
