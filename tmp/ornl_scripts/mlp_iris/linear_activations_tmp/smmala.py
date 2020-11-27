# %% Import packages

import os
import csv

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.distributions import Normal

from eeyore.data import Iris
from eeyore.stats import softabs
from eeyore.models import mlp
from eeyore.mcmc import SMMALA

from timeit import default_timer as timer
from datetime import timedelta

import matplotlib.pyplot as plt

# %% Avoid issuing memory warning due to number of plots

# plt.rcParams.update({'figure.max_open_warning': 0})

# %% Set up output directory

OUTDIR = os.path.join("output", "smmala", "single_chain", "normal_prior", "var003")

if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

# %% Load iris data

iris = Iris(dtype=torch.float64)
dataloader = DataLoader(iris, batch_size=150)

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=[4, 3, 3], activations=[lambda x: 0.001 * x, None])
model = mlp.MLP(
    hparams=hparams, loss=lambda x, y: nn.CrossEntropyLoss(reduction='sum')(x, torch.argmax(y, 1)), dtype=torch.float64)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(3)*torch.ones(model.num_params(), dtype=model.dtype))

# %% Set number of chains, iterations, burnin iterations and proposal scale

num_iterations = 1100
num_burnin = 100
num_post_burnin = num_iterations - num_burnin

drift_step = 0.05

# %% Setup SMMALA sampler

theta0 = model.prior.sample()
sampler = SMMALA(model, theta0, dataloader, step=drift_step, transform=lambda hessian: softabs(hessian, a=1000.))

# %% Run MALA sampler

start_time = timer()

sampler.run(num_iterations=num_iterations, num_burnin=num_burnin)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Compute acceptance rate

print("Acceptance rate: {}".format(sampler.chain.acceptance_rate()))

# %% Plot traces of simulated chain

for i in range(model.num_params()):
    plt.figure()
    plt.plot(sampler.chain.get_theta(i))

# %% Save simulated Markov chain in file

# Store all parameters in a single torch tensor
chain = torch.empty(num_post_burnin, model.num_params())
for i in range(model.num_params()):
    chain[:, i] = torch.tensor(sampler.chain.get_theta(i))

# Save tensor in file
with open(os.path.join(OUTDIR, "chain.csv"), 'w') as file:
    np.savetxt(file, chain.cpu().detach().numpy(), delimiter=',', newline='\n', header='')

## Save acceptance diagnostic for simulated Markov chain

with open(os.path.join(OUTDIR, "accepted.txt"), 'w') as file:
    writer = csv.writer(file)
    for a in sampler.chain.vals['accepted']:
        writer.writerow([a])

## Save runtime of MC simulation

with open(os.path.join(OUTDIR, "runtime.txt"), 'w') as file:
    file.write(str("Runtime: {}".format(timedelta(seconds=end_time-start_time))))
    file.write("\n")
