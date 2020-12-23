# %% Import packages

from bnn_mcmc_examples.datasets import load_xydataset_from_file
from bnn_mcmc_examples.datasets.noisy_xor.data2.constants import training_data_path
from bnn_mcmc_examples.mlp.noisy_xor.setting2.constants import dtype

# %% Load dataloader

_, dataloader = load_xydataset_from_file(training_data_path, dtype=dtype)
