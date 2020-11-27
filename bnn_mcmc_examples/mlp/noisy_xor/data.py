# %% Import packages

from torch.utils.data import DataLoader

from eeyore.datasets import XYDataset

from bnn_mcmc_examples.datasets import data_paths

from .constants import dtype

# %% Load XOR data

# xor = XYDataset.from_eeyore('xor', dtype=dtype)
noisy_xor = XYDataset.from_file(path=data_paths['noisy_xor'], dtype=dtype)
dataloader = DataLoader(noisy_xor, batch_size=len(noisy_xor))
