## Import packages

import os
import csv

import numpy as np

import torch

from torch.utils.data import DataLoader
from torch.distributions import Normal

from eeyore.data import XOR
from eeyore.models import mlp
from eeyore.kernels import NormalTransitionKernel
from eeyore.mcmc import MetropolisHastings

from timeit import default_timer as timer
from datetime import timedelta

## Load XOR data

xor = XOR(dtype=torch.float64)
dataloader = DataLoader(xor, batch_size=4)

## Setup MLP model

hparams = mlp.Hyperparameters(dims=[2, 2, 1])
model = mlp.MLP(hparams=hparams, dtype=torch.float64)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(3)*torch.ones(model.num_params(), dtype=model.dtype)
)

## Setup Metropolis-Hastings sampler

theta0 = model.prior.sample()
num_params = model.num_params()
kernel = NormalTransitionKernel(torch.zeros(num_params), np.sqrt(1.55)*torch.ones(num_params), dtype=torch.float64)
sampler = MetropolisHastings(model, theta0, dataloader, kernel)

## Run Metropolis-Hastings sampler

start_time = timer()

sampler.run(num_iterations=110000, num_burnin=10000)

end_time = timer()
print("Runtime: {}".format(timedelta(seconds=end_time-start_time)))

## Compute acceptance rate

print("Acceptance rate: {}".format(sampler.chain.acceptance_rate()))

## Save simulated Markov chain in file

for i in range(model.num_params()):
    chain = sampler.chain.get_theta(i)
    with open(os.path.join("output", str("mh_chain{:02d}.txt".format(i+1))), 'w') as file:
        writer = csv.writer(file)
        for state in chain:
            writer.writerow([state])

## Save acceptance diagnostic for simulated Markov chain

with open(os.path.join("output", "mh_accepted.txt"), 'w') as file:
    writer = csv.writer(file)
    for a in sampler.chain.vals['accepted']:
        writer.writerow([a])

## Save runtime of MC simulation

with open(os.path.join("output", "mh_runtime.txt"), 'w') as file:
    file.write(str("Runtime: {}".format(timedelta(seconds=end_time-start_time))))
    file.write("\n")
