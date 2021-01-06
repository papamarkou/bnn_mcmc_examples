# %% Load packages

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import kanga.plots as ps
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

print(pred_interval_y)

# %%

# fig, (ax, cbar_ax) = plt.subplots(2, figsize=[7, 7], gridspec_kw=None)

sns.heatmap(
    pred_interval_y,
    cmap=redblue_cmap,
    # linewidths=0.01,
    # linecolor='white',
    cbar=True,
    cbar_kws=None,
    # cbar_ax=cbar_ax,
    square=True,
    # xticklabels=pred_interval_x1,
    # yticklabels=yticklabels,
    mask=None
)
#    , gridspec_kw=None, vmin=-1, vmax=1, cmap=redblue_cmap, linewidths=0.01,
#    linecolor='white', cbar=True, cbar_kws=None, square=True, xticklabels=None, yticklabels=None, mask=None, upper=True,
#    xtick_labelsize=8, ytick_rotation=0, ytick_labelsize=8, cbar_labelsize=8, fname=None, quality=100, transparent=True,
#    bbox_inches='tight', pad_inches=0.1
# )

plt.ylim(0, 10)
