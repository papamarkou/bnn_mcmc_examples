# %% Import packages

from torch.utils.data import DataLoader

from eeyore.datasets import XYDataset

from .constants import dtype

# %% Load XOR data

xor = XYDataset.from_eeyore('xor', dtype=dtype)
dataloader = DataLoader(xor, batch_size=len(xor))
