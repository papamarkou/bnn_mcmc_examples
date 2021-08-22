# %% Import packages

import torch
import torchvision.datasets as datasets

from torch.nn.functional import one_hot

from eeyore.datasets import XYDataset

from bnn_mcmc_examples.datasets import data_root
from bnn_mcmc_examples.examples.mlp.mnist.constants import dtype

# %% Create training dataset

training_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=None)

training_dataset = XYDataset(training_dataset.data.to(dtype), training_dataset.targets.to(dtype))

training_dataset.x = \
    (training_dataset.x - torch.mean(training_dataset.x, dim=0, keepdim=True))/ \
    torch.std(training_dataset.x, dim=0, keepdim=True, unbiased=False)

training_dataset.y = one_hot(training_dataset.y.squeeze(-1).long()).to(training_dataset.y.dtype)

# %% Create test dataset

test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=None)

test_dataset = XYDataset(test_dataset.data.to(dtype), test_dataset.targets.to(dtype))
