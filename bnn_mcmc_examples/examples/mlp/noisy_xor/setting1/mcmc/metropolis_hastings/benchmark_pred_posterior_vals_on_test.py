# %% Load packages

import numpy as np

from eeyore.chains import ChainLists

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.constants import dtype
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.constants import num_chains, pred_iter_thres
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.metropolis_hastings.constants import sampler_output_run_paths
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.model import model
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.datascanners import test_dataloader

# %% Load chain lists

chain_lists = ChainLists.from_file(sampler_output_run_paths, keys=['sample'], dtype=dtype)

# %% Drop burn-in samples

for i in range(num_chains):
    chain_lists.vals['sample'][i] = chain_lists.vals['sample'][i][pred_iter_thres:]

# %% Generate ground truch

pred_posterior = np.empty([num_chains, len(test_dataloader)])

for k in range(num_chains):
    print(k)
    for i, (x, y) in enumerate(test_dataloader):
        pred_posterior[k, i] = model.predictive_posterior(chain_lists.vals['sample'][k], x.squeeze(), y.squeeze(-1))

# %% Save predictive posteriors

for i in range(num_chains):
    np.savetxt(sampler_output_run_paths[i].joinpath('pred_posterior_on_test.csv'), pred_posterior[i], delimiter=',')
