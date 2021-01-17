# %% Load packages

import numpy as np
import torch

from eeyore.chains import ChainLists

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.constants import dtype
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.constants import (
    num_chains, pred_interval_x1, pred_interval_x2, pred_iter_thres
)
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.metropolis_hastings.constants import sampler_output_run_paths
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.model import model

# %% Load chain lists

chain_lists = ChainLists.from_file(sampler_output_run_paths, keys=['sample'], dtype=dtype)

# %% Drop burn-in samples

for i in range(num_chains):
    chain_lists.vals['sample'][i] = chain_lists.vals['sample'][i][pred_iter_thres:]

# %% Generate ground truch

pred_posterior_yhat = np.empty([num_chains, len(pred_interval_x1), len(pred_interval_x2)])

for k in range(num_chains):
    for i in range(len(pred_interval_x1)):
        for j in range(len(pred_interval_x2)):
            pred_posterior_yhat[k, i, j] = model.predictive_posterior(
                chain_lists.vals['sample'][k],
                torch.tensor([pred_interval_x1[i], pred_interval_x2[j]], dtype=dtype),
                torch.tensor([1.], dtype=dtype)
            )

# %% Save predictive posteriors

for i in range(num_chains):
    np.savetxt(sampler_output_run_paths[i].joinpath('pred_posterior_on_grid.csv'), pred_posterior_yhat[i], delimiter=',')
