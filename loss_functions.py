import torch


def loss_function_empirical_integral(inputs, outputs, rho, gamma, f, c=torch.pnorm(2)):
    res_lambda = outputs['lambda'] * rho
    inner_sum_h = outputs['h'].sum(dim=0)
    res_h = inner_sum_h.mean()
    res_g = outputs['g'].mean()
    inner_arg = f(inputs['y'][:,]) - inner_sum_h - outputs['lambda']*c(inputs['x'][:,], inputs['y'][:,]) - outputs['g']
    inner_arg_with_beta_gamma = gamma * torch.square(torch.max(0, inner_arg))
    res_beta_gamma = torch.mean(inner_arg_with_beta_gamma)
    loss = res_lambda + res_h + res_g + res_beta_gamma
    return loss
