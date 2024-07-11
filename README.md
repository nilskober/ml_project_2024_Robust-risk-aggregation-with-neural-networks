# Machine Learning Study Project
### Based on "Robust risk aggregation with neural networks" (Eckstein et al., 2020)

This repository contains the code for our study project in the context of the lecture "Stochastics of Machine Learning" in the summer term 2024 at the University Freiburg by Prof. Dr. Thorsten Schmidt.
It implements the optimization problem described in the paper "Robust risk aggregation with neural networks" by Eckstein et al. (2020) using neural networks.

## Running the code
This project requires Python 3.12 or higher. It has only been tested on MacOS and Linux with Python 3.12.4. We recommend installing Python via Anaconda or Miniconda.
After having installed a suitable Python version, make sure to install the required packages by running the following command:
```pip install -r requirements.txt```

Start the optimization by executing the following command (replace `example_4_1` with the name of the config file):

```python run_optimization.py --config-name example_4_1```

We have implemented the following config files:
- `example_4_1` (Example in section 4.1 from the paper)
- `example_4_2` (Example in section 4.2 from the paper)
- `example_gaussian` (our own example implementing robustly estimating the Avarage Value at Risk of a sum of the coordinates of a multivariate Gaussian distribution) 

You can also modify parameters directly in the command line by adding them to the command. For example, to change the number of epochs to 100000, run the following command:

```python run_optimization.py --config-name example_4_1 num_epochs_total=100000```

To run larger scale experiments, you can also use the --multirun option to run multiple experiments in parallel. For example, to run Example 4.1. with different seeds and different values for rho, execute the following command:

```python run_optimization.py --config-name example_4_1 --multirun seed=1,2,3,4,5 rho=0.1,0.2,0.3,0.4,0.5```

The results will be stored in the folder structure as shown in the following example (with an additional folder for the alpha value in case of `example_4_2` and `example_gaussian`):
```experiments/results/example_4_1_YYYY-MM-DD_HH-MM-SS/rho=0.5/gamma=1280/batch_size=128/width=128/depth=3/seed=0```

This structure can be adjusted as needed in the config files. The test and training trajectories are stored in the `results_test.csv` and `results_train.csv` files respectively.

The results can be visualized using the functions `plot_fixed_epoch()` and `plot_epochs()` in `plots.py`. For example usage, please refer to `run_plots.py`. Please make sure to use it accurately, as it is not yet fully tested and error messages might not be very informative. 

## Work in progress / known issues:
- `example_4_1_HPO` is a config file for hyperparameter optimization using the optuna plugin for hydra. It is not yet fully implemented and tested. In particular, the folder structure needs to account for all hyperparameters that are optimized to save the individual trials.
- In each optimization step, the batch is sampled from the given distribution. It would probably be more efficient to sample (a very large batch) once, save that and then reuse it across different experiments.
- In our tests, the optimization is faster on a CPU than on a GPU. This might be due to the sampling step being the bottleneck. We do not know the exact reason yet.
- The plotting functions might not work as expected if not all results are available for the specified parameters.

## Authors
Anja Buschle, Simon Ebert, Nils Kober and Lukas Riepl

## License
GNU General Public License v3.0 or later

See COPYING.md to see the full text.
