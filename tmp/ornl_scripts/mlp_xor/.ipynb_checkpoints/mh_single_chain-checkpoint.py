## Import packages

import os
import csv

import numpy as np

import torch
import torch.nn.functional as F

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
model = mlp.MLP(hparams=hparams, dtype=torch.float64, loss=F.mse_loss)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(100)*torch.ones(model.num_params(), dtype=model.dtype)
)
print(model.num_params())

## Set number of chains, iterations, burnin iterations and proposal scale

num_iterations = 110000
num_burnin = 10000
num_post_burnin = num_iterations - num_burnin

proposal_scale = np.sqrt(1.55)

## Setup Metropolis-Hastings sampler

theta0 = model.prior.sample()
num_params = model.num_params()
kernel = NormalTransitionKernel(torch.zeros(num_params), proposal_scale*torch.ones(num_params), dtype=torch.float64)
sampler = MetropolisHastings(model, theta0, dataloader, kernel)

## Run Metropolis-Hastings sampler

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
with open(os.path.join("output", "mh", "single_chain", "chain.csv"), 'w') as file:
    np.savetxt(file, chain.cpu().detach().numpy(), delimiter=',', newline='\n', header='')

## Save acceptance diagnostic for simulated Markov chain

with open(os.path.join("output", "mh", "single_chain", "accepted.txt"), 'w') as file:
    writer = csv.writer(file)
    for a in sampler.chain.vals['accepted']:
        writer.writerow([a])

## Save runtime of MC simulation

with open(os.path.join("output", "mh", "single_chain", "runtime.txt"), 'w') as file:
    file.write(str("Runtime: {}".format(timedelta(seconds=end_time-start_time))))
    file.write("\n")