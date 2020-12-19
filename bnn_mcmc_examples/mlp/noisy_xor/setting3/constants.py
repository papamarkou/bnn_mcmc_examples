# %% Import packages

import torch

from pathlib import Path

# %% Define constants

output_path = Path.home().joinpath('output', 'bnn_mcmc_examples', 'mlp', 'noisy_xor', 'setting3')

num_features = 2

mlp_dims = [num_features, 2, 1]

dtype = torch.float32
batch_size = 50

num_chains = 4

num_epochs = 1100
num_burnin_epochs = 0
verbose = True
verbose_step = 1000
