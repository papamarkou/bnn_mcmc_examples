# %% Import packages

import numpy as np
import torch

from pathlib import Path

from bnn_mcmc_examples.datasets import data_paths

# %% Define constants

data_name = 'data1'
data_path = data_paths['noisy_xor'].joinpath(data_name)
output_path = Path.home().joinpath('output', 'bnn_mcmc_examples', 'data', 'noisy_xor', data_name)

num_classes = 4
num_samples = np.repeat(125, num_classes)

dtype = torch.float32
