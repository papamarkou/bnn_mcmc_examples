# %% Import packages

from torch.utils.data import DataLoader

from eeyore.datasets import XYDataset

from bnn_mcmc_examples.datasets import data_paths
from bnn_mcmc_examples.mlp.noisy_xor.setting1.constants import dtype, setting_name

# %% Load XOR data

noisy_xor = XYDataset.from_file(path=data_paths['noisy_xor'].joinpath(setting_name), dtype=dtype)
dataloader = DataLoader(noisy_xor, batch_size=len(noisy_xor))
