# %% Load packages

import numpy as np

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.constants import ground_truth_x1, ground_truth_x2

# %%

ground_truth_y = np.empty([len(ground_truth_x1), len(ground_truth_x2)])

for x1 in ground_truth_x1:
    for x2 in ground_truth_x2:
        print(x1, x2)
