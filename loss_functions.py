from functools import partial

import torch


def loss_function_empirical_integral(inputs, outputs, additional_parameters, rho, gamma, f, input_dim, c=partial(torch.norm, p=1, dim=1)):
    res_lambda = additional_parameters['lambda'] * rho
    inner_sum_h = outputs['h'].sum(dim=1)
    res_h = inner_sum_h.mean()
    res_g = outputs['g'].mean()
    # Split inputs into x and y, based on the size of the input
    inputs_x = inputs[:, :input_dim]
    inputs_y = inputs[:, input_dim:]
    inner_arg = f(inputs_y) - inner_sum_h - additional_parameters['lambda']*c(inputs_x - inputs_y) - outputs['g'].reshape((-1,))
    inner_arg_with_beta_gamma = gamma * torch.square(torch.max(torch.tensor(0), inner_arg))
    res_beta_gamma = torch.mean(inner_arg_with_beta_gamma)
    loss = res_lambda + res_h + res_g + res_beta_gamma
    return loss


def loss_function_empirical_integral_avar(inputs, outputs, additional_parameters, rho, gamma, f, input_dim, c=partial(torch.norm, p=1, dim=1)):
    res_lambda = additional_parameters['lambda'] * rho
    inner_sum_h = outputs['h'].sum(dim=1)
    res_h = inner_sum_h.mean()
    res_g = outputs['g'].mean()
    # Split inputs into x and y, based on the size of the input
    inputs_x = inputs[:, :input_dim]
    inputs_y = inputs[:, input_dim:]
    inner_arg = f(inputs_y, additional_parameters['tau']) - inner_sum_h - additional_parameters['lambda'] * c(inputs_x - inputs_y) - outputs[
        'g'].reshape((-1,))
    inner_arg_with_beta_gamma = gamma * torch.square(torch.max(torch.tensor(0), inner_arg))
    res_beta_gamma = torch.mean(inner_arg_with_beta_gamma)
    loss = res_lambda + res_h + res_g + res_beta_gamma
    return loss
