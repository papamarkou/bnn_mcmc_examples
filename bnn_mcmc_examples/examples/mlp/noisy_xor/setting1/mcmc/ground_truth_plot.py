# %% Load packages

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from kanga.plots import redblue_cmap

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.constants import pred_interval_x1, pred_interval_x2

# %%

pred_interval_y = np.empty([len(pred_interval_x1), len(pred_interval_x2)])

for i in range(len(pred_interval_x1)):
    for j in range(len(pred_interval_x2)):
        if pred_interval_x1[i] < 0.5:
            pred_interval_y[i, j] = 0 if (pred_interval_x2[j] < 0.5) else 1
        else:
            pred_interval_y[i, j] = 1 if (pred_interval_x2[j] < 0.5) else 0

# %%

xticks = np.linspace(0, len(pred_interval_x1)-1, num=6, dtype=np.int)
xticklabels = [np.round(pred_interval_x1[idx], decimals=2) for idx in xticks]

yticks = np.linspace(0, len(pred_interval_x2)-1, num=6, dtype=np.int)
yticklabels = [np.round(pred_interval_x2[idx], decimals=2) for idx in yticks]

ax = sns.heatmap(
    pred_interval_y,
    cmap=redblue_cmap,
    linewidths=0.01,
    linecolor='white',
    cbar=True,
    square=True
)

plt.ylim(0, len(pred_interval_x2))

ax.set_xticks(xticks+0.5)
ax.set_xticklabels(xticklabels, fontsize=8)

ax.set_yticks(yticks+0.5)
ax.set_yticklabels(yticklabels, rotation=0, fontsize=8)

ax.collections[0].colorbar.ax.tick_params(labelsize=8)
