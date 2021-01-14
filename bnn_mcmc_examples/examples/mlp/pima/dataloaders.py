# %% Import packages

from torch.utils.data import DataLoader

from bnn_mcmc_examples.examples.mlp.pima.datasets import training_dataset, test_dataset

# %% Create training dataloader

training_dataloader = DataLoader(training_dataset)

# %% Create test dataloader

test_dataloader = DataLoader(test_dataset)
