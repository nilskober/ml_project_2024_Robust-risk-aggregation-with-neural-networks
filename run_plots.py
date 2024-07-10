from functools import partial
from math import sqrt

from plots import plot_fixed_epoch, plot_epoch
import os

base_dir = 'experiments/results/'
base_output_dir = 'experiments/plots/'

# create the output directory if it does not exist

os.makedirs(base_output_dir, exist_ok=True)

# Example 4.1 analytical solution
def f_ref_4_1(rho, fixed_params):
    return (1.0 + min(0.5, rho))/2.0

# Example 4.2 analytical solution
def f_ref_4_2_upper(rho, fixed_params, alpha):
    return min(1+alpha, 2-2.0/3.0 * sqrt(2 - 2*alpha) + rho/(2-2*alpha))

def f_ref_4_2_lower(rho, fixed_params, alpha):
    return min(1+alpha, 2-2.0/3.0 * sqrt(2 - 2*alpha) + (2*(-3+2*sqrt(2 - 2*alpha) + 3*alpha)*rho)/(3*(2-alpha)*(1-alpha)*alpha))


# Arguments for the run_plots.py script
args_fixed_epoch = [
    {
        'root_dir': base_dir + 'group=example_4_1_rho',
        'x_param': 'rho',
        'fixed_params': {'gamma': 1280, 'depth': 3, 'width': 128, 'batch_size': 128},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_rho.png',
        'title': 'Example 4.1 - Varying rho',
        'upper_bound_func': f_ref_4_1,
        'lower_bound_func': f_ref_4_1
    },
    {
        'root_dir': base_dir + 'group=example_4_1_gamma',
        'x_param': 'gamma',
        'fixed_params': {'rho': 0.5, 'depth': 3, 'width': 128, 'batch_size': 128},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_gamma_all.png',
        'title': 'Example 4.1 - Varying gamma',
    },
    {
        'root_dir': base_dir + 'group=example_4_1_gamma',
        'x_param': 'gamma',
        'fixed_params': {'rho': 0.5, 'depth': 3, 'width': 128, 'batch_size': 128},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_gamma_part.png',
        'title': 'Example 4.1 - Varying gamma',
        'x_param_start': 1000,
        'x_param_end': 30000,
    },
    {
        'root_dir': base_dir + 'group=example_4_1_gamma',
        'x_param': 'gamma',
        'fixed_params': {'rho': 0.5, 'depth': 3, 'width': 128, 'batch_size': 128},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_gamma_part_short.png',
        'title': 'Example 4.1 - Varying gamma',
        'x_param_start': 500,
        'x_param_end': 5000,
    },
    {
        'root_dir': base_dir + 'group=example_4_1_width',
        'x_param': 'width',
        'fixed_params': {'rho': 0.5, 'depth': 3, 'gamma': 1280, 'batch_size': 128},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_width.png',
        'title': 'Example 4.1 - Varying width',
    },
    {
        'root_dir': base_dir + 'group=example_4_1_depth',
        'x_param': 'depth',
        'fixed_params': {'rho': 0.5, 'gamma': 1280, 'width': 128, 'batch_size': 128},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_depth.png',
        'title': 'Example 4.1 - Varying depth',
    },
    {
        'root_dir': base_dir + 'group=example_4_1_batch_size',
        'x_param': 'batch_size',
        'fixed_params': {'rho': 0.5, 'gamma': 1280, 'width': 128, 'depth': 3},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_batch_size.png',
        'title': 'Example 4.1 - Varying batch size',
    },
    #
    # Example 4.2
    {
        'root_dir': base_dir + 'group=example_4_2_rho',
        'x_param': 'rho',
        'fixed_params': {'gamma': 1280, 'depth': 3, 'width': 128, 'alpha': 0.7},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_2_rho.png',
        'title': 'Example 4.2 - Varying rho',
        'upper_bound_func': partial(f_ref_4_2_upper, alpha=0.7),
        'lower_bound_func': partial(f_ref_4_2_lower, alpha=0.7),
    }
]


for a in args_fixed_epoch:
    print(f"Running with args: {a}")
    plot_fixed_epoch(**a)
    print("Done")



args_plot_epochs = [
    {
        'root_dir': base_dir + 'group=example_4_1_epochs',
        'fixed_params': {'gamma': 1280, 'depth': 3, 'width': 128, 'rho': 0.5, 'batch_size': 128},
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_epochs_all.png',
        'title': 'Example 4.1 - Varying epochs',
    },
    {
        'root_dir': base_dir + 'group=example_4_1_epochs',
        'fixed_params': {'gamma': 1280, 'depth': 3, 'width': 128, 'rho': 0.5, 'batch_size': 128},
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_epochs_part_large.png',
        'title': 'Example 4.1 - Varying epochs',
        'epoch_start': 10000,
        'epoch_end': 100000,
    },
    {
        'root_dir': base_dir + 'group=example_4_1_epochs',
        'fixed_params': {'gamma': 1280, 'depth': 3, 'width': 128, 'rho': 0.5, 'batch_size': 128},
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_epochs_part_small.png',
        'title': 'Example 4.1 - Varying epochs',
        'epoch_start': 15000,
        'epoch_end': 25000,
    }
]
for a in args_plot_epochs:
    print(f"Running with args: {a}")
    plot_epoch(**a)
    print("Done")