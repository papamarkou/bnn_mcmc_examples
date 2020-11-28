# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from bnn_mcmc_examples.mlp.noisy_xor.data import noisy_xor
from bnn_mcmc_examples.mlp.noisy_xor.constants import num_classes, num_samples

# %% Plot noisy XOR points

num_samples_cumsum = np.hstack((0, num_samples)).cumsum()

cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i in range(num_classes):
    plt.plot(
        noisy_xor.x[num_samples_cumsum[i]:num_samples_cumsum[i+1], 0],
        noisy_xor.x[num_samples_cumsum[i]:num_samples_cumsum[i+1], 1],
        'o',
        color=cols[i],
        marker='o',
        markersize=3
    )
