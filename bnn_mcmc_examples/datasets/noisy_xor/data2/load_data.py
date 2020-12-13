# %% Import packages

import torch

from torch.utils.data import DataLoader

from eeyore.datasets import XYDataset

from bnn_mcmc_examples.datasets.noisy_xor.data2.constants import data_path

# %% Function for loading XOR data

def load_data(dtype=torch.float32):
    noisy_xor = XYDataset.from_file(path=data_path, dtype=dtype)
    dataloader = DataLoader(noisy_xor, batch_size=len(noisy_xor))
    return noisy_xor, dataloader
