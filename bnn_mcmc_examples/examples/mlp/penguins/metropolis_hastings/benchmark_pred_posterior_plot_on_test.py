# %% Load packages

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bnn_mcmc_examples.examples.mlp.penguins.constants import num_chains
from bnn_mcmc_examples.examples.mlp.penguins.dataloaders import test_dataloader
from bnn_mcmc_examples.examples.mlp.penguins.metropolis_hastings.constants import sampler_output_run_paths

# %% Load test labels

_, test_labels = next(iter(test_dataloader))
test_labels = torch.argmax(test_labels, 1).detach().cpu().numpy()

# %% Plot predictive posteriors

pred_colors = {'correct': '#bcbd22', 'wrong': '#d62728'}
# '#bcbd22': rio grande, similar to yellow green
# ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

patch_list = []
for key in pred_colors:
    patch_list.append(mpatches.Patch(color=pred_colors[key], label=key))

legend_patches = [mpatches.Patch(color=pred_colors[key], label=key.capitalize()) for key in pred_colors]

for i in range(num_chains):
    test_pred_df = pd.read_csv(
        sampler_output_run_paths[i].joinpath('pred_posterior_on_test.txt'),
        header=None,
        names=['class0', 'class1', 'class2']
    )

    test_pred_df['preds'] = np.loadtxt(
        sampler_output_run_paths[i].joinpath('preds_via_bm.txt'), dtype=np.int, delimiter=',', skiprows=0
    )

    test_pred_df['labels'] = test_labels

    test_pred_df.sort_values(['labels'], ascending=True, inplace=True)

    test_pred_df = pd.concat([
        test_pred_df.loc[test_pred_df['labels'] == 0].sort_values(['class0'], ascending=True),
        test_pred_df.loc[test_pred_df['labels'] == 1].sort_values(['class1'], ascending=True),
        test_pred_df.loc[test_pred_df['labels'] == 2].sort_values(['class2'], ascending=True)
    ])

    test_pred_df['color'] = [
        pred_colors['correct'] if cmp else pred_colors['wrong'] for cmp in test_pred_df['preds'] == test_pred_df['labels']
    ]

    plt.figure(figsize=[8, 4])

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

    plt.vlines(
        x=range(len(test_labels)),
        ymin=0,
        # ymax=test_pred_df['class0'], # This should be a vector of class0, class1, class2
        color=test_pred_df['color'],
        linewidth=2
    )

    #plt.bar(
    #    range(len(test_labels)),
    #    np.hstack([test_logit_dict[0], test_logit_dict[1]]),
    #    width=0.7,
    #    color=bar_colors,
    #    align='edge'
    #)

    plt.legend(handles=legend_patches, loc='upper left', ncol=1)

    plt.axhline(y=0.5, xmin=0, xmax=len(test_labels), color='black', linestyle='dashed', linewidth=1.5)

    plt.axvline(x=0.5*len(test_labels), ymin=0, ymax=1, color='black', linestyle='dotted', linewidth=1.5)

    plt.savefig(
        sampler_output_run_paths[i].joinpath('pred_posterior_on_test.png'),
        pil_kwargs={'quality': 100},
        transparent=True,
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.close()
