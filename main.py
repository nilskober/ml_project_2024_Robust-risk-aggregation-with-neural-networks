import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import numpy as np

from data_loader import data_generator_from_distribution
from optimization_pipeline import optimize_model


@hydra.main(config_path="configs", config_name="example_4_1")
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

    # optimize the model
    optimize_model(
        device=device,
        model=model,
        loss_function=cfg.loss_function,
        data_loader=data_gen,
        loss_params={
            'rho': cfg.rho,
            'gamma': cfg.gamma,
            'f': hydra.utils.instantiate(cfg.f),
            'input_dim': cfg.input_dim
        },
        num_epochs_total=cfg.num_epochs_total,
        start_decay_at=cfg.start_decay_at,
        lr=cfg.lr,
        print_every=cfg.print_every
    )

    # Save model to disk
    torch.save(model.state_dict(), cfg.model_save_path)

    # Test model
    print("Test model")
    model.eval()
    data_gen_test = data_generator_from_distribution(cfg.batch_size_testing, distribution)
    data_test = next(data_gen_test)
    inputs = data_test.to(device)
    outputs = model(inputs)
    loss = cfg.loss_function(inputs, outputs, cfg.rho, cfg.gamma, cfg.f, cfg.input_dim)
    print(f'Loss on test data: {loss.item():.4f}')


if __name__ == "__main__":
    main()
