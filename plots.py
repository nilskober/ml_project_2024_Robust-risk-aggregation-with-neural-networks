import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_parameters_from_path(path, root_dir):
    params = {}
    relative_path = os.path.relpath(path, root_dir)
    parts = relative_path.split(os.sep)
    for part in parts:
        if '=' in part:
            key, value = part.split('=')
            params[key] = float(value) if '.' in value else int(value)
    return params


def collect_data(root_dir):
    data = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'results_test.csv':
                file_path = os.path.join(subdir, file)
                params = extract_parameters_from_path(subdir, root_dir)
                df = pd.read_csv(file_path)
                df['file_path'] = file_path
                df['params'] = [params] * len(df)
                data.append(df)
    return pd.concat(data, ignore_index=True)


def filter_by_seeds(data, fixed_params, num_seeds=None):
    filtered_data = data[data['params'].apply(lambda p: all(p.get(k) == v for k, v in fixed_params.items()))]
    unique_seeds = filtered_data['params'].apply(lambda p: p['seed']).unique()

    if num_seeds is not None:
        if num_seeds > len(unique_seeds):
            print(f"Error: Requested {num_seeds} seeds, but only {len(unique_seeds)} available.")
            return pd.DataFrame()  # Return an empty DataFrame to handle error gracefully
        selected_seeds = unique_seeds[:num_seeds]
        filtered_data = filtered_data[filtered_data['params'].apply(lambda p: p['seed']).isin(selected_seeds)]

    return filtered_data


def plot_type1(data, x_param, fixed_params, epoch, num_seeds=None, x_param_start=None, x_param_end=None, title=None):
    filtered_data = filter_by_seeds(data, fixed_params, num_seeds)
    if filtered_data.empty:
        return

    epoch_data = filtered_data[filtered_data['epoch'] == epoch]

    # Filter by specified x_param range if provided
    if x_param_start is not None:
        epoch_data = epoch_data[epoch_data['params'].apply(lambda p: p[x_param]) >= x_param_start]
    if x_param_end is not None:
        epoch_data = epoch_data[epoch_data['params'].apply(lambda p: p[x_param]) <= x_param_end]

    # Group by x_param and calculate mean and standard error
    grouped = epoch_data.groupby(epoch_data['params'].apply(lambda p: p[x_param])).agg(
        {'loss': ['mean', 'std', 'count']}).reset_index()
    grouped.columns = [x_param, 'mean_loss', 'std_loss', 'count']
    grouped['stderr_loss'] = grouped['std_loss'] / np.sqrt(grouped['count'])

    plt.figure(figsize=(10, 6))
    plt.plot(grouped[x_param], grouped['mean_loss'], marker='o', label='Mean Loss')
    plt.fill_between(grouped[x_param], grouped['mean_loss'] - grouped['stderr_loss'],
                     grouped['mean_loss'] + grouped['stderr_loss'], alpha=0.3, label='Standard Error')

    plt.xlabel(x_param)
    plt.ylabel('Loss')

    if title is None:
        title = f'Loss vs {x_param} at epoch {epoch}'
    plt.title(title)

    # Create the legend with additional information
    settings_text = (f"Fixed Parameters: {fixed_params}\n"
                     f"Seeds: {num_seeds}\n"
                     f"Epoch: {epoch}\n")
    if x_param_start is not None or x_param_end is not None:
        settings_text += f"{x_param} range: {x_param_start}-{x_param_end}"

    plt.legend(loc='upper right', title=settings_text)
    plt.grid(True)
    plt.show()


def plot_type2(data, fixed_params, num_seeds=None, log_scale=False, epoch_start=None, epoch_end=None):
    filtered_data = filter_by_seeds(data, fixed_params, num_seeds)
    if filtered_data.empty:
        return

    # Filter by specified epoch range if provided
    if epoch_start is not None:
        filtered_data = filtered_data[filtered_data['epoch'] >= epoch_start]
    if epoch_end is not None:
        filtered_data = filtered_data[filtered_data['epoch'] <= epoch_end]

    # Group by epoch and calculate mean and standard error
    grouped = filtered_data.groupby('epoch').agg({'loss': ['mean', 'std', 'count']}).reset_index()
    grouped.columns = ['epoch', 'mean_loss', 'std_loss', 'count']
    grouped['stderr_loss'] = grouped['std_loss'] / np.sqrt(grouped['count'])

    plt.figure(figsize=(10, 6))
    plt.plot(grouped['epoch'], grouped['mean_loss'], marker='o', label='Mean Loss')
    plt.fill_between(grouped['epoch'], grouped['mean_loss'] - grouped['stderr_loss'],
                     grouped['mean_loss'] + grouped['stderr_loss'], alpha=0.3, label='Standard Error')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = f'Loss vs Epoch with fixed parameters {fixed_params} and {num_seeds} seeds'
    if epoch_start is not None or epoch_end is not None:
        title += f' (epoch range: {epoch_start}-{epoch_end})'
    plt.title(title)
    if log_scale:
        plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage:
root_dir = 'experiments/results/group=example_4_1_rho_2024-07-03-19-02-49'  # Change this to the directory where your data is stored
# data = collect_data(root_dir)

# Plot Type 1 Example
# x_param = 'rho'  # Change this to the parameter you want to plot on the x-axis
# fixed_params = {'gamma': 1280, 'width': 128, 'depth': 3}
# epoch = 100000  # Specify the epoch to fix
# num_seeds = 11  # Specify the number of seeds to use
# plot_type1(data, x_param, fixed_params, epoch, num_seeds)
#

root_dir = 'experiments/results/group=example_4_1_gamma_2024-07-04-12-39-44'
data = collect_data(root_dir)
x_param = 'gamma'  # Change this to the parameter you want to plot on the x-axis
fixed_params = {'rho': 0.5, 'width': 128, 'depth': 3}
epoch = 100000  # Specify the epoch to fix
num_seeds = 11  # Specify the number of seeds to use
plot_type1(data, x_param, fixed_params, epoch, num_seeds)


# # # Plot Type 2 Example
# fixed_params = {'rho': 0.3, 'gamma': 1280, 'width': 128, 'depth': 3}
# num_seeds = 4  # Specify the number of seeds to use
# plot_type2(data, fixed_params, num_seeds, log_scale=False, epoch_start=15000)
