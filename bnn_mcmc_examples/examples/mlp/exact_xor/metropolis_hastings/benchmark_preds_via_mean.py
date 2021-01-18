# %% Load packages

import numpy as np

from eeyore.chains import ChainLists

from bnn_mcmc_examples.examples.mlp.exact_xor.constants import diagnostic_iter_thres, dtype, num_chains
from bnn_mcmc_examples.examples.mlp.exact_xor.dataloader import dataloader
from bnn_mcmc_examples.examples.mlp.exact_xor.metropolis_hastings.constants import sampler_output_run_paths
from bnn_mcmc_examples.examples.mlp.exact_xor.model import model

# %% Load chain lists

chain_lists = ChainLists.from_file(sampler_output_run_paths, keys=['sample'], dtype=dtype)

# %% Drop burn-in samples

for i in range(num_chains):
    chain_lists.vals['sample'][i] = chain_lists.vals['sample'][i][diagnostic_iter_thres:]

# %% Load test data and labels

data, labels = next(iter(dataloader))

# %% Compute chain means

means = chain_lists.mean()

# %% Make and save predictions

for i in range(num_chains):
    # Initialize model parameters
    model.set_params(means[i, :].clone().detach())

    # Compute logits
    logits = model(data)

    # Make predictions
    preds = logits.squeeze() > 0.5

    # Save predictions
    np.savetxt(sampler_output_run_paths[i].joinpath('preds_via_mean.txt'), preds, fmt='%d', delimiter=',')
