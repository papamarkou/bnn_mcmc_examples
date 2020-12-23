# %% Import packages

import torch

from pathlib import Path

# %% Define constants

output_path = Path.home().joinpath('output', 'bnn_mcmc_examples', 'mlp', 'noisy_xor', 'setting3')

num_features = 2

mlp_dims = [num_features, 2, 1]

dtype = torch.float32

num_chains = 4

num_mcmc_epochs = 1100
num_mcmc_burnin_epochs = 0
mcmc_batch_size = 50

verbose = True
mcmc_verbose_step = 1000
