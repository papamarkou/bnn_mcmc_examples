# %% Import packages

import numpy as np
import torch

from pathlib import Path

from bnn_mcmc_examples.datasets import data_paths

# %% Define constants

setting_name = 'setting1'
data_path = data_paths['noisy_xor'].joinpath(setting_name)
output_path = Path.home().joinpath('output', 'bnn_mcmc_examples', 'mlp', 'noisy_xor', setting_name)

num_classes = 4
num_features = 2
num_samples = np.repeat(250, num_classes)

mlp_dims = [num_features, 2, 1]

dtype = torch.float32

num_chains = 4

num_epochs = 1100
num_burnin_epochs = 100
verbose = True
verbose_step = 100
