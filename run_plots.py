from plots import plot_fixed_epoch

base_dir = 'experiments/results/'
base_output_dir = 'experiments/plots/'

# Arguments for the run_plots.py script
args_fixed_epoch = [
    {
        'root_dir': base_dir + 'group=example_4_1_rho',
        'x_param': 'rho',
        'fixed_params': {'gamma': 1280, 'depth': 3, 'width': 128},
        'epoch': 20000,
        'num_seeds': 11,
        'output_path': base_output_dir + 'example_4_1_rho.png',
    }
]


for a in args_fixed_epoch:
    print(f"Running with args: {a}")
    plot_fixed_epoch(**a)
    print("Done")
