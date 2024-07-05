from plots import plot_fixed_epoch, plot_epoch
import os

base_dir = 'experiments/results/'
base_output_dir = 'experiments/plots/'

# create the output directory if it does not exist

os.makedirs(base_output_dir, exist_ok=True)

# Arguments for the run_plots.py script
args_fixed_epoch = [
    {
        'root_dir': base_dir + 'group=example_4_1_rho',
        'x_param': 'rho',
        'fixed_params': {'gamma': 1280, 'depth': 3, 'width': 128},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_rho.png',
        'title': 'Example 4.1 - Varying rho',
    },
    {
        'root_dir': base_dir + 'group=example_4_1_gamma',
        'x_param': 'gamma',
        'fixed_params': {'rho': 0.5, 'depth': 3, 'width': 128},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_gamma_all.png',
        'title': 'Example 4.1 - Varying gamma',
    },
    {
        'root_dir': base_dir + 'group=example_4_1_gamma',
        'x_param': 'gamma',
        'fixed_params': {'rho': 0.5, 'depth': 3, 'width': 128},
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
        'fixed_params': {'rho': 0.5, 'depth': 3, 'width': 128},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_gamma_part_short.png',
        'title': 'Example 4.1 - Varying gamma',
        'x_param_start': 1000,
        'x_param_end': 5000,
    },
    {
        'root_dir': base_dir + 'group=example_4_1_size',
        'x_param': 'width',
        'fixed_params': {'rho': 0.5, 'depth': 3, 'gamma': 1280},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_width.png',
        'title': 'Example 4.1 - Varying width',
    },
    {
        'root_dir': base_dir + 'group=example_4_1_size',
        'x_param': 'depth',
        'fixed_params': {'rho': 0.5, 'gamma': 1280, 'width': 128},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_depth.png',
        'title': 'Example 4.1 - Varying depth',
    },
    {
        'root_dir': base_dir + 'group=example_4_1_batchsize',
        'x_param': 'batchsize',
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
        'epoch': 19999,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_2_rho.png',
        'title': 'Example 4.2 - Varying rho',
    },
    # Example 3
    {
        'root_dir': base_dir + 'group=example_3_rho',
        'x_param': 'rho',
        'fixed_params': {'gamma': 1280, 'depth': 3, 'width': 128, 'alpha': 0.7},
        'epoch': 19999,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_3_rho.png',
        'title': 'Example 3 - Varying rho',
    },
]


for a in args_fixed_epoch:
    print(f"Running with args: {a}")
    plot_fixed_epoch(**a)
    print("Done")



args_plot_epochs = [
    {
        'root_dir': base_dir + 'group=example_4_1_rho',
        'fixed_params': {'gamma': 1280, 'depth': 3, 'width': 128, 'rho': 0.5},
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_epochs_all.png',
        'title': 'Example 4.1 - Varying epochs',
    },
    {
        'root_dir': base_dir + 'group=example_4_1_rho',
        'fixed_params': {'gamma': 1280, 'depth': 3, 'width': 128, 'rho': 0.5},
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_epochs_part_large.png',
        'title': 'Example 4.1 - Varying epochs',
        'epoch_start': 10000,
        'epoch_end': 100000,
    },
    {
        'root_dir': base_dir + 'group=example_4_1_rho',
        'fixed_params': {'gamma': 1280, 'depth': 3, 'width': 128, 'rho': 0.5},
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