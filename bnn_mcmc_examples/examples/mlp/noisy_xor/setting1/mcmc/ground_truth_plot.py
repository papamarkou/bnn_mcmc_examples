# %% Load packages

import numpy as np

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.constants import ground_truth_x1, ground_truth_x2

# %%

ground_truth_y = np.empty([len(ground_truth_x1), len(ground_truth_x2)])

for i in range(len(ground_truth_x1)):
    for j in range(len(ground_truth_x2)):
        if ground_truth_x1[i] < 0.5:
            ground_truth_y[i, j] = 0 if (ground_truth_x2[j] < 0.5) else 1
        else:
            ground_truth_y[i, j] = 1 if (ground_truth_x2[j] < 0.5) else 0

print(ground_truth_y)
