# %% Load packages

import numpy as np

from sklearn.metrics import accuracy_score

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.constants import num_chains
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.dataloaders import test_dataloader
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.hmc.constants import (
    sampler_output_path, sampler_output_run_paths
)

# %% Load test data and labels

_, test_labels = next(iter(test_dataloader))

# %% Compute predictive accuracies

accuracies = np.empty(num_chains)

for i in range(num_chains):
    test_preds = np.loadtxt(sampler_output_run_paths[i].joinpath('preds_via_mean.txt'), skiprows=0)

    accuracies[i] = accuracy_score(test_preds, test_labels.squeeze())

# %% Save predictive accuracies

np.savetxt(sampler_output_path.joinpath('accuracies_via_mean.txt'), accuracies)
