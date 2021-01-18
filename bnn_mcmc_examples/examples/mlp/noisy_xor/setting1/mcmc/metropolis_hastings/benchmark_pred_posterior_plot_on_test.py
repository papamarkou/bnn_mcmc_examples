# %% Load packages

import matplotlib.pyplot as plt
import numpy as np
import torch

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.constants import num_chains
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.dataloaders import test_dataloader
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.mcmc.metropolis_hastings.constants import sampler_output_run_paths

# %% Load test data and labels

test_data, test_labels = next(iter(test_dataloader))

# %% Plot predictive posteriors

pred_colors = {'correct': 'green', 'wrong': 'red'}

num_correct_0s = sum(test_logit_dict[0] < 0.5)
num_correct_1s = sum(test_logit_dict[1] >= 0.5)
bar_colors = num_correct_0s * [pred_colors['correct']]
bar_colors.extend((len(test_logit_dict[0]) - num_correct_0s) * [pred_colors['wrong']])
bar_colors.extend(num_correct_1s * [pred_colors['correct']])
bar_colors.extend((len(test_logit_dict[1]) - num_correct_1s) * [pred_colors['wrong']])

['green']

for i in range(num_chains):    
    # Load test logits
    test_logits = np.loadtxt(sampler_output_run_paths[i].joinpath('pred_posterior_on_test.txt'), delimiter=',', skiprows=0)

    # Construct dictionary of test logits
    test_logit_dict = {
        0: np.sort(test_logits[(1 - test_labels.squeeze(-1)).to(dtype=torch.bool)]),
        1: np.sort(test_logits[test_labels.squeeze(-1).to(dtype=torch.bool)])[::-1]
    }

    num_correct_0s = sum(test_logit_dict[0] < 0.5)
    num_correct_1s = sum(test_logit_dict[1] >= 0.5)
    bar_colors = num_correct_0s * [pred_colors['correct']]
    bar_colors.extend((len(test_logit_dict[0]) - num_correct_0s) * [pred_colors['wrong']])
    bar_colors.extend(num_correct_1s * [pred_colors['correct']])
    bar_colors.extend((len(test_logit_dict[1]) - num_correct_1s) * [pred_colors['wrong']])

    plt.bar(
        range(len(test_labels)),
        np.hstack([test_logit_dict[0], test_logit_dict[1]]),
        color=bar_colors
    )

    plt.savefig(
        sampler_output_run_paths[i].joinpath('pred_posterior_on_test.png'),
        pil_kwargs={'quality': 100},
        transparent=True,
        bbox_inches='tight',
        pad_inches=0.1
    )

    # Save predictions
    plt.close()
