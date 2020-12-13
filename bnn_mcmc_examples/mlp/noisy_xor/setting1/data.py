# %% Import packages

from torch.utils.data import DataLoader

from eeyore.datasets import XYDataset

from bnn_mcmc_examples.mlp.noisy_xor.generate_data1 import data_path
from bnn_mcmc_examples.mlp.noisy_xor.setting1.constants import dtype

# %% Load XOR data

noisy_xor = XYDataset.from_file(path=data_path, dtype=dtype)
dataloader = DataLoader(noisy_xor, batch_size=len(noisy_xor))
