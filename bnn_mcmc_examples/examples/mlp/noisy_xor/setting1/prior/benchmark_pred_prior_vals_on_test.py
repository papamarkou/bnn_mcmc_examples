# %% Load packages

import numpy as np
import torch

from eeyore.chains import ChainLists

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.constants import dtype
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.constants import num_chains, pred_iter_thres
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.datascanners import test_dataloader
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.model import model
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.prior.constants import sampler_output_run_paths

# %% Load chain lists

chain_lists = ChainLists.from_file(sampler_output_run_paths, keys=['sample'], dtype=dtype)

# %% Drop burn-in samples

for i in range(num_chains):
    chain_lists.vals['sample'][i] = chain_lists.vals['sample'][i][pred_iter_thres:]

# %% Compute and save predictive priors

verbose_msg = 'Evaluating predictive prior based on chain {:' \
    + str(len(str(num_chains))) \
    + '} out of ' \
    + str(num_chains) \
    + ' at test point {:' \
    + str(len(str(len(test_dataloader)))) \
    + '} out of ' \
    + str(len(test_dataloader)) \
    + '...'

for k in range(num_chains):
    test_pred_probs = np.empty([len(test_dataloader), 2])

    for i, (x, _) in enumerate(test_dataloader):
        print(verbose_msg.format(k+1, i+1))

        test_pred_probs[i, 1] = model.predictive_posterior(
            chain_lists.vals['sample'][k], x, torch.tensor([[1.]], dtype=dtype)
        ).item()
        test_pred_probs[i, 0] = 1. - test_pred_probs[i, 1]

    np.savetxt(sampler_output_run_paths[k].joinpath('pred_prior_on_test.csv'), test_pred_probs, delimiter=',')
