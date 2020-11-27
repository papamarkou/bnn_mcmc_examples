# %% Import packages

import numpy as np

import torch

from torch.utils.data import DataLoader
from torch.distributions import Normal

from eeyore.data import XOR
from eeyore.models import mlp
from eeyore.mcmc import MALA
from eeyore.mcmc import ChainFile

from timeit import default_timer as timer
from datetime import timedelta

# %% Load XOR data

xor = XOR(dtype=torch.float64)
dataloader = DataLoader(xor, batch_size=4)

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=[2, 2, 1])
model = mlp.MLP(hparams=hparams, dtype=torch.float64)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(3)*torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Setup chain

chain = ChainFile(keys=['theta', 'target_val', 'grad_val', 'accepted'])

# %% Set number of chains, iterations, burnin iterations and proposal scale

num_iterations = 110
num_burnin = 10
num_post_burnin = num_iterations - num_burnin

drift_step = 1.74

# %% Setup MALA sampler

theta0 = model.prior.sample()
sampler = MALA(model, theta0, dataloader, step=drift_step, chain=chain)

# %% Run MALA sampler

start_time = timer()

sampler.run(num_iterations=num_iterations, num_burnin=num_burnin)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))
