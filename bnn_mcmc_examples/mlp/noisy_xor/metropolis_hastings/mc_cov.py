# %% Load packages

import numpy as np

from kanga.chains import ChainArrays

from bnn_mcmc_examples.mlp.noisy_xor.constants import num_chains
from bnn_mcmc_examples.mlp.noisy_xor.metropolis_hastings.constants import sampler_output_path, sampler_output_run_paths

# %% Load chain arrays from file

chain_arrays = ChainArrays.from_file(sampler_output_run_paths, keys=['sample'])

# %% Compute Monte Carlo covariance matrix

mc_cov_mats = chain_arrays.mc_cov()

# %% Save Monte Carlo covariance matrices to files

for i in range(num_chains):
    np.savetxt(sampler_output_run_paths[i].joinpath('mc_cov.csv'), mc_cov_mats[i], delimiter=',')

# %% Save mean of Monte Carlo covariance matrices

np.savetxt(sampler_output_path.joinpath('mean_mc_cov.csv'), mc_cov_mats.mean(0), delimiter=',')

# %% Compute Monte Carlo correlation matrix

mc_cor_mats = chain_arrays.mc_cor(mc_cov_mat=mc_cov_mats)

# %% Save Monte Carlo correlation matrices to files

for i in range(num_chains):
    np.savetxt(sampler_output_run_paths[i].joinpath('mc_cor.csv'), mc_cor_mats[i], delimiter=',')

# %% Save mean of Monte Carlo correlation matrices

np.savetxt(sampler_output_path.joinpath('mean_mc_cor.csv'), mc_cor_mats.mean(0), delimiter=',')
