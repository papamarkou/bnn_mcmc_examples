# %% Import packages

import torch

from pathlib import Path

# %% Define constants

output_path = Path.home().joinpath('output', 'bnn_mcmc_examples', 'mlp', 'noisy_xor')

input_size = 2
mlp_dims = [input_size, 2, 1]

dtype = torch.float32

num_chains = 4

num_epochs = 11000
num_burnin_epochs = 1000
verbose = True
verbose_step = 1000
