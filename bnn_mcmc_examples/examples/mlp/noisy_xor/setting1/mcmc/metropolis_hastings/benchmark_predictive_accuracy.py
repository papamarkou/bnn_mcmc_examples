# %% Load packages

import numpy as np

from sklearn.metrics import accuracy_score

from eeyore.chains import ChainLists

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.constants import dtype
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.metropolis_hastings.constants import (
    sampler_output_path, sampler_output_run_paths
)
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.model import model
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.optim.dataloaders import test_dataloader

# %%

chain_lists = ChainLists.from_file(sampler_output_run_paths, keys=['sample'], dtype=dtype)

# %%

for i in range(chain_lists.num_chains()):
    chain_lists.vals['sample'][i] = chain_lists.vals['sample'][i][10000:]

# %% Load test data and labels

test_data, test_labels = next(iter(test_dataloader))

# %%

means = chain_lists.mean()

# %%

for i in range(chain_lists.num_chains()):
    # Initialize model parameters
    model.set_params(means[i, :])

    # Compute test logits
    test_logits = model(test_data)

    # Make test predictions
    test_preds = test_logits.squeeze() > 0.5

    # Compute test accuracy
    print(accuracy_score(test_preds, test_labels.squeeze()))

# %%
