import logging
from os.path import join

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os

import loss_functions
from data_loader import data_generator_from_distribution
from optimization_pipeline import optimize_model

logger = logging.getLogger("run_optimization")
logger_hydra = logging.getLogger("hydra_multirun")

@hydra.main(config_path="configs", version_base="1.2")
def main(cfg: DictConfig) -> None:
    task_id = HydraConfig.get().job.num
    logger_hydra.info(f"Starting task {task_id + 1}")

    # print the config
    logger.info(OmegaConf.to_yaml(cfg))
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
            }
        )
    # choose the device (cpu or gpu)
    if cfg.use_gpu:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device("cpu")
    logger.info(f"device: {device}")
    model.to(device)

    # instantiate the distribution from the config
    distribution = hydra.utils.instantiate(cfg.distribution)
    data_gen = data_generator_from_distribution(cfg.batch_size_training, distribution)
    data_gen_test = data_generator_from_distribution(cfg.batch_size_testing, distribution)

    # instantiate the function f from the config
    f = hydra.utils.instantiate(cfg.target_function)
    # loss_function = loss_functions.loss_function_empirical_integral
    loss_function = hydra.utils.instantiate(cfg.loss_function)
    # optimize the model
    res = optimize_model(
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
        print_every=cfg.print_every,
        test_every=cfg.test_every,
        test_data_loader=data_gen_test
    )

    # Save model to disk
    if cfg.save_model:
        # check if file exists
        if not os.path.exists(cfg.model_save_path) or cfg.overwrite_model:
            try:
                torch.save(model.state_dict(), cfg.model_save_path)
                with open(cfg.model_save_path + '_params', 'w') as f:
                    f.write("name,param\n")
                    for p in additional_parameters:
                        f.write(f"{p['name']},{p['param'].item()}\n")
            except FileNotFoundError:
                logger.info(f"Could not save model to {cfg.model_save_path}")
            # Save additional parameters to disk

        else:
            logger.info(f"Model file {cfg.model_save_path} already exists. Set overwrite_model to True to overwrite.")

    # Save train results to disk
    if cfg.save_results:
        # check if file exists
        if not os.path.exists(cfg.train_results_save_path) or cfg.overwrite_results:
            try:
                with open(cfg.train_results_save_path, 'w') as f:
                    f.write("epoch,loss\n")
                    for epoch, loss in res['loss_trajectory_train']:
                        f.write(f"{epoch},{loss}\n")
            except FileNotFoundError:
                logger.info(f"Could not save train results to {cfg.train_results_save_path}")
        else:
            logger.info(f"Train results file {cfg.train_results_save_path} already exists. Set overwrite_train_results to True to overwrite.")
        # check if file exists
        if not os.path.exists(cfg.test_results_save_path) or cfg.overwrite_results:
            try:
                with open(cfg.test_results_save_path, 'w') as f:
                    f.write("epoch,loss\n")
                    for epoch, loss in res['loss_trajectory_test']:
                        f.write(f"{epoch},{loss}\n")
            except FileNotFoundError:
                logger.info(f"Could not save test results to {cfg.test_results_save_path}")
        else:
            logger.info(
                f"Test results file {cfg.test_results_save_path} already exists. Set overwrite_test_results to True to overwrite.")

    # Test model
    # print("Test model")
    # model.eval()
    # data_gen_test = data_generator_from_distribution(cfg.batch_size_testing, distribution)
    # data_test = next(data_gen_test)
    # inputs = data_test.to(device)
    # outputs = model(inputs)
    # additional_parameters_values = {p['name']: p['param'] for p in additional_parameters}
    # loss = loss_function(inputs, outputs, additional_parameters_values, cfg.rho, cfg.gamma, f, cfg.input_dim)
    # print(f'Loss on test data: {loss.item():.4f}')

    logger_hydra.info(f"Completed task {task_id + 1}")
    # return difference to analytical solution if available
    if hasattr(cfg, 'analytical_solution'):
        loss = res['loss_trajectory_test'][-1][1]
        return abs(loss - cfg.analytical_solution)

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    root_directory_str = str(os.path.normpath(dirname))
    os.environ["HYDRA_ROOT"] = root_directory_str
    main()