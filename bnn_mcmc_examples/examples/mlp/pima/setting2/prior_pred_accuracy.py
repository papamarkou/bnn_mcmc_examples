# %% Load packages

import numpy as np
import torch

from sklearn.metrics import accuracy_score

from bnn_mcmc_examples.examples.mlp.pima.setting2.constants import num_chains, output_path
from bnn_mcmc_examples.examples.mlp.pima.setting2.dataloaders import test_dataloader
from bnn_mcmc_examples.examples.mlp.pima.setting2.model import model

# %% Create output directory if it does not exist

output_path.mkdir(parents=True, exist_ok=True)

# %% Load test data and labels

test_data, test_labels = next(iter(test_dataloader))

# %% Draw samples from the prior

prior_samples = torch.stack([model.prior.sample() for i in range(num_chains)])

# %% Compute predictive accuracies

accuracies = np.empty(num_chains)

for i in range(num_chains):
    # Initialize model parameters
    model.set_params(prior_samples[i, :].clone().detach())

    # Compute test logits
    test_logits = model(test_data)

    # Make test predictions
    test_preds = test_logits.squeeze() > 0.5

    # Compute test accuracy
    accuracies[i] = accuracy_score(test_preds, test_labels.squeeze())

# %% Save prior samples

np.savetxt(output_path.joinpath('prior_samples.csv'), prior_samples.detach().cpu().numpy(), delimiter=',')

# %% Save predictive accuracies

np.savetxt(output_path.joinpath('prior_accuracy.txt'), accuracies)
