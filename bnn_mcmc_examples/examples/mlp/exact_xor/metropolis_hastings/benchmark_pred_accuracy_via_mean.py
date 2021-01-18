# %% Load packages

import numpy as np

from sklearn.metrics import accuracy_score

from bnn_mcmc_examples.examples.mlp.exact_xor.constants import num_chains
from bnn_mcmc_examples.examples.mlp.exact_xor.dataloader import dataloader
from bnn_mcmc_examples.examples.mlp.exact_xor.metropolis_hastings.constants import (
    sampler_output_path, sampler_output_run_paths
)

# %% Load test data and labels

data, labels = next(iter(dataloader))

# %% Compute predictive accuracies

accuracies = np.empty(num_chains)

for i in range(num_chains):
    # Load predictions
    preds = np.loadtxt(sampler_output_run_paths[i].joinpath('preds_via_mean.txt'), delimiter=',', skiprows=0)

    # Compute accuracy
    accuracies[i] = accuracy_score(preds, labels.squeeze())

# %% Save predictive accuracies

np.savetxt(sampler_output_path.joinpath('accuracies_via_mean.txt'), accuracies)
