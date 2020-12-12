# %% Load packages

import numpy as np

import kanga.plots as ps

from bnn_mcmc_examples.mlp.noisy_xor.setting1.metropolis_hastings.constants import sampler_output_path

# %% Load chain arrays and correlation matrix

mean_mc_cor_mat = np.loadtxt(sampler_output_path.joinpath('mean_mc_cor.csv'), delimiter=',', skiprows=0)

# %% Plot heat map of mean correlation matrix

ps.cor_heatmap(mean_mc_cor_mat, fname=sampler_output_path.joinpath('mean_mc_cor.png'))
