# %% Import packages

import numpy as np

from pathlib import Path

from bnn_mcmc_examples.datasets import data_paths

# %% Define constants

data_name = 'data2'
data_path = data_paths['noisy_xor'].joinpath(data_name)
output_path = Path.home().joinpath('output', 'bnn_mcmc_examples', 'data', 'noisy_xor', data_name)

num_classes = 4
num_samples = np.repeat(1250, num_classes)
