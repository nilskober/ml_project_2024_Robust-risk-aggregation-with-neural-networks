import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import numpy as np

import loss_functions
from data_loader import data_generator_from_distribution
from optimization_pipeline import optimize_model


@hydra.main(config_path="configs", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # print the config
    print(OmegaConf.to_yaml(cfg))
    # set the seed in torch and numpy
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.mps.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # instantiate the model from the config
    model = hydra.utils.instantiate(cfg.model)

    # read additional parameters from config
    additional_parameters = [
        {
            'name': 'lambda',
            'param': torch.tensor([cfg.lambda_par.initial], requires_grad=True),
            'update_every': cfg.lambda_par.update_every,
            'start_optimizing_at': cfg.lambda_par.start_optimizing_at,
            'initial_lr': cfg.lambda_par.initial_lr,
            'start_decay_at': cfg.lambda_par.start_decay_at,
            'decay_every': cfg.lambda_par.decay_every,
            'decay_rate': cfg.lambda_par.decay_rate,
            'lower_bound': cfg.lambda_par.lower_bound
        }
    ]
    # check if there is a second parameter tau
    if cfg.tau_par is not None:
        additional_parameters.append(
            {
                'name': 'tau',
                'param': torch.tensor([cfg.tau_par.initial], requires_grad=True),
                'update_every': cfg.tau_par.update_every,
                'start_optimizing_at': cfg.tau_par.start_optimizing_at,
                'initial_lr': cfg.tau_par.initial_lr,
                'start_decay_at': cfg.tau_par.start_decay_at,
                'decay_every': cfg.tau_par.decay_every,
                'decay_rate': cfg.tau_par.decay_rate,
                'lower_bound': cfg.tau_par.lower_bound,
                'alpha': cfg.tau_par.alpha
            }
        )
    # choose the device (cpu or gpu)
    if cfg.use_gpu:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device("cpu")
    print("device:", device)
    model.to(device)

    # instantiate the distribution from the config
    distribution = hydra.utils.instantiate(cfg.distribution)
    data_gen = data_generator_from_distribution(cfg.batch_size_training, distribution)

    # instantiate the function f from the config
    f = hydra.utils.instantiate(cfg.target_function)
    # loss_function = loss_functions.loss_function_empirical_integral
    loss_function = hydra.utils.instantiate(cfg.loss_function)
    # optimize the model
    optimize_model(
        device=device,
        model=model,
        additional_parameters=additional_parameters,
        loss_function=loss_function,
        data_loader=data_gen,
        loss_params={
            'rho': cfg.rho,
            'gamma': cfg.gamma,
            'f': f,
            'input_dim': cfg.input_dim
        },
        num_epochs_total=cfg.num_epochs_total,
        start_decay_at=cfg.start_decay_at,
        lr=cfg.lr,
        print_every=cfg.print_every
    )

    # Save model to disk
    if cfg.save_model:
        # check if file exists
        import os
        if not os.path.exists(cfg.model_save_path) or cfg.overwrite_model:
            try:
                torch.save(model.state_dict(), cfg.model_save_path)
            except FileNotFoundError:
                print(f"Could not save model to {cfg.model_save_path}")
        else:
            print(f"Model file {cfg.model_save_path} already exists. Set overwrite_model to True to overwrite.")



    # Test model
    print("Test model")
    model.eval()
    data_gen_test = data_generator_from_distribution(cfg.batch_size_testing, distribution)
    data_test = next(data_gen_test)
    inputs = data_test.to(device)
    outputs = model(inputs)
    additional_parameters_values = {p['name']: p['param'] for p in additional_parameters}
    loss = loss_function(inputs, outputs, additional_parameters_values, cfg.rho, cfg.gamma, f, cfg.input_dim)
    print(f'Loss on test data: {loss.item():.4f}')


if __name__ == "__main__":
    main()
