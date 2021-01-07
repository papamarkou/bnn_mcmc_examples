# %% Load packages

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from kanga.plots import redblue_cmap

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.constants import num_chains, pred_interval_x1, pred_interval_x2
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.metropolis_hastings.constants import sampler_output_run_paths

# %% Load predictive posteriors

num_chains = 2
pred_posterior_yhat = []
for i in range(num_chains):
    pred_posterior_yhat.append(
        np.loadtxt(sampler_output_run_paths[i].joinpath('pred_posterior_yhat.csv'), delimiter=',', skiprows=0)
    )
pred_posterior_yhat = np.stack(pred_posterior_yhat)

# %% Plot heat maps of predictive posteriors

for i in range(num_chains):
    num_ticks = 4

    xticks = np.linspace(0, len(pred_interval_x1)-1, num=num_ticks, dtype=np.int)
    xticklabels = [np.round(pred_interval_x1[idx], decimals=2) for idx in xticks]

    yticks = np.linspace(0, len(pred_interval_x2)-1, num=num_ticks, dtype=np.int)
    yticklabels = [np.round(pred_interval_x2[idx], decimals=2) for idx in yticks]

    ax = sns.heatmap(
        pred_posterior_yhat[i, :, :],
        cmap=redblue_cmap,
        linewidths=0.01,
        linecolor='white',
        cbar=True,
        square=True
    )

    plt.ylim(0, len(pred_interval_x2))

    ax.set_xticks(xticks+0.5)
    ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)

    ax.set_yticks(yticks+0.5)
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=8)

    ax.collections[0].colorbar.ax.tick_params(labelsize=8)

    plt.savefig(
        sampler_output_run_paths[i].joinpath('pred_posterior_yhat.png'),
        pil_kwargs={'quality': 100},
        transparent=True,
        bbox_inches='tight',
        pad_inches=0.1
    )
