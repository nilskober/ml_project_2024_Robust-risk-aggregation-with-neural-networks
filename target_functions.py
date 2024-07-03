import torch


# used in example 4.1
def f_max(x):
    return torch.max(x[:, 0], x[:, 1])


# for example 4.2

#ToDo: Load the Parameter alpha from example_4_2.yaml instead of hardcoding it as 0.7
def f_avar(x, tau):
    #return torch.mul(torch.max(x[:, 0] + x[:, 1] - tau, 0)[0], 1 / (1 - 0.7))+tau
    return tau[:] + torch.mul(torch.max(x[:, 0] + x[:, 1] - tau[:], 0)[0], torch.tensor(1 / (1 - 0.7)))
