# %% Load packages

# predictive_posterior_y
# predictive_posterior_yhat

import numpy as np

from sklearn.metrics import accuracy_score

from eeyore.chains import ChainLists

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.constants import dtype
# from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.constants import diagnostic_iter_thres, num_chains
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.constants import num_chains
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.metropolis_hastings.constants import (
    sampler_output_path, sampler_output_run_paths
)
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.model import model
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.optim.dataloaders import test_dataloader

# %% Load chain lists

chain_lists = ChainLists.from_file(sampler_output_run_paths, keys=['sample'], dtype=dtype)

# %% Drop burn-in samples

diagnostic_iter_thres = 100000
for i in range(num_chains):
    chain_lists.vals['sample'][i] = chain_lists.vals['sample'][i][diagnostic_iter_thres:]

# %% Load test data and labels

test_data, test_labels = next(iter(test_dataloader))

# %%

i = 40
print(test_data[i], test_labels[i])
tmp01 = model.predictive_posterior(chain_lists.vals['sample'][9], test_data[i], test_labels[i])
tmp02 = model.predictive_posterior(means, test_data[i], test_labels[i])
print(tmp01, tmp02)
