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

## Set up output directory

OUTDIR = os.path.join("output", "smmala", "multiple_chains", "normal_prior", "var003")

if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

# %% Load IRIS data

iris = Iris()
dataloader = DataLoader(iris, batch_size=150)

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=[4, 3, 3], activations=[torch.sigmoid, None])
model = mlp.MLP(
    hparams=hparams, loss=lambda x, y: nn.CrossEntropyLoss(reduction='sum')(x, torch.argmax(y, 1)), dtype=torch.float64)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(3)*torch.ones(model.num_params(), dtype=model.dtype))

# %% Set number of chains, iterations, burnin iterations and MALA drift step

num_chains = 5
num_iterations = 110 # 110000
num_burnin = 10 # 10000
num_post_burnin = num_iterations - num_burnin

drift_step = 0.02

msg = "Run {:" + str(len(str(num_chains))) + "}, duration {}, acceptance rate {}"

# %% Run simulation and save output

for i in range(num_chains):
    # Setup MALA sampler
    theta0 = model.prior.sample()
    sampler = SMMALA(model, theta0, dataloader, step=drift_step, transform=lambda hessian: softabs(hessian, a=1000.))

    # Run MALA sampler
    start_time = timer()
    sampler.run(num_iterations=num_iterations, num_burnin=num_burnin)
    end_time = timer()
    print(msg.format(i+1, timedelta(seconds=end_time-start_time), sampler.chain.acceptance_rate()))

    # Store all parameters in a single torch tensor
    chain = torch.empty(num_post_burnin, model.num_params())
    for j in range(model.num_params()):
        chain[:, j] = torch.tensor(sampler.chain.get_theta(j))

    # Save tensor in file
    with open(os.path.join(OUTDIR, str("chain{:02d}.csv".format(i+1))), 'w') as file:
        np.savetxt(file, chain.cpu().detach().numpy(), delimiter=',', newline='\n', header='')

    ## Save acceptance diagnostic for simulated Markov chain
    with open(os.path.join(OUTDIR, str("accepted{:02d}.txt".format(i+1))), 'w') as file:
        writer = csv.writer(file)
        for a in sampler.chain.vals['accepted']:
            writer.writerow([a])

    ## Save runtime of simulation
    with open(os.path.join(OUTDIR, str("runtime{:02d}.txt".format(i+1))), 'w') as file:
        file.write(str("Runtime: {}".format(timedelta(seconds=end_time-start_time))))
        file.write("\n")
