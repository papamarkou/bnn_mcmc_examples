## Import packages

import os
import csv

import numpy as np

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.distributions import Normal
from torch import nn

from eeyore.data import XOR
from eeyore.models import mlp
from eeyore.kernels import NormalTransitionKernel
from eeyore.mcmc import MALA

from timeit import default_timer as timer
from datetime import timedelta

## Load XOR data

xor = XOR(dtype=torch.float64)
dataloader = DataLoader(xor, batch_size=4)

## Setup MLP model

hparams = mlp.Hyperparameters(dims=[2, 3, 1]) #, activations=[F.relu, torch.sigmoid])
model = mlp.MLP(hparams=hparams, dtype=torch.float64) # , loss=nn.BCELoss) #, loss=F.mse_loss)
print(model.num_params())
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(3)*torch.ones(model.num_params(), dtype=model.dtype)
)

## Set number of chains, iterations, burnin iterations and proposal scale

num_iterations = 110000
num_burnin = 10000
num_post_burnin = num_iterations - num_burnin

drift_step = 1.74

## Setup MALA sampler

theta0 = model.prior.sample()
sampler = MALA(model, theta0, dataloader, step=drift_step)

## Run MALA sampler

start_time = timer()

sampler.run(num_iterations=num_iterations, num_burnin=num_burnin)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

## Compute acceptance rate

print("Acceptance rate: {}".format(sampler.chain.acceptance_rate()))

## Save simulated Markov chain in file

# Store all parameters in a single torch tensor
chain = torch.empty(num_post_burnin, model.num_params())
for i in range(model.num_params()):
    chain[:, i] = torch.tensor(sampler.chain.get_theta(i))

# Save tensor in file
with open(os.path.join("output", "mala", "single_chain", "chain.csv"), 'w') as file:
    np.savetxt(file, chain.cpu().detach().numpy(), delimiter=',', newline='\n', header='')

## Save acceptance diagnostic for simulated Markov chain

with open(os.path.join("output", "mala", "single_chain", "accepted.txt"), 'w') as file:
    writer = csv.writer(file)
    for a in sampler.chain.vals['accepted']:
        writer.writerow([a])

## Save runtime of MC simulation

with open(os.path.join("output", "mala", "single_chain", "runtime.txt"), 'w') as file:
    file.write(str("Runtime: {}".format(timedelta(seconds=end_time-start_time))))
    file.write("\n")
