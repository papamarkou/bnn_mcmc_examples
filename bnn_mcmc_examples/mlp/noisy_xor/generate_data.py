# %% Load packages

import numpy as np

from bnn_mcmc_examples.datasets import data_paths, simulate_xor
from bnn_mcmc_examples.mlp.noisy_xor.constants import num_samples

# %% Simulate noisy XOR points

x, y = simulate_xor(n=num_samples)

# %% Save simulated noisy XOR points

np.savetxt(data_paths['noisy_xor'].joinpath('x.csv'), x, delimiter=',', header='x1,x2', comments='')

np.savetxt(data_paths['noisy_xor'].joinpath('y.csv'), y, fmt='%d', header='y', comments='')
