# %% Import packages

import os
import csv

import numpy as np

import torch

from torch.utils.data import DataLoader
from torch.distributions import Normal

from eeyore.data import Iris
from eeyore.models import mlp
from eeyore.mcmc import MALA

from timeit import default_timer as timer
from datetime import timedelta

# %% Set up output directory

OUTDIR = os.path.join("output", "mala", "multiple_chains", "normal_prior", "mean10")

if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

# %% Load IRIS data

iris = Iris()
dataloader = DataLoader(iris, batch_size=150)

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=[4, 3, 3])
model = mlp.MLP(hparams=hparams, dtype=torch.float64)
model.prior = Normal(
    10*torch.ones(model.num_params(), dtype=model.dtype),
    np.sqrt(3)*torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Set number of chains, iterations, burnin iterations and MALA drift step

num_chains = 4
num_iterations = 11000
num_burnin = 1000
num_post_burnin = num_iterations - num_burnin

drift_step = 0.0001

msg = "Run {:" + str(len(str(num_chains))) + "}, duration {}, acceptance rate {}"

# %% Run simulation and save output

for i in [0]: # range(num_chains):
    # Setup MALA sampler
    theta0 = model.prior.sample()
    sampler = MALA(model, theta0, dataloader, step=drift_step)

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
