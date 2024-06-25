import torch


def loss_function(inputs, outputs, rho, gamma, f, c):
    res_lambda = outputs['lambda'] * rho
    sum_h = outputs['h'].sum(dim=0)
    inner_arg = f(inputs['y'][:,]) - sum_h - outputs['lambda']*c(inputs['x'][:,],inputs['y'][:,]) - outputs['g']
    approx_integral = torch.mean(inner_arg)


