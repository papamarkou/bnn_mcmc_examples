# %% Import packages

import numpy as np
import torch

from pathlib import Path

# %% Define constants

output_path = Path.home().joinpath('output', 'bnn_mcmc_examples', 'mlp', 'noisy_xor')

num_classes = 4
num_features = 2
num_samples = np.repeat(250, num_classes)

mlp_dims = [num_features, 2, 1]

dtype = torch.float32

num_chains = 4

num_epochs = 11000
num_burnin_epochs = 1000
verbose = True
verbose_step = 1000
