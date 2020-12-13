# %% Load packages

import numpy as np

from bnn_mcmc_examples.datasets import data_paths, simulate_xor

# %% Set and create data directory if it does not exist

data_path = data_paths['noisy_xor'].joinpath('data1')
data_path.mkdir(parents=True, exist_ok=True)

# %% Set number of data points

num_classes = 4
num_samples = np.repeat(1250, num_classes)

# %% Simulate noisy XOR points

x, y = simulate_xor(n=num_samples)

# %% Save simulated noisy XOR points

np.savetxt(data_path.joinpath('x.csv'), x, delimiter=',', header='x1,x2', comments='')
np.savetxt(data_path.joinpath('y.csv'), y, fmt='%d', header='y', comments='')
