#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:51:18 2019

@author: 9tp
"""

# %% Import packages

import os

import numpy as np

import torch

from torch.utils.data import DataLoader
from torch.distributions import Normal

from eeyore.data import Iris
from eeyore.models import mlp
from eeyore.mcmc import PowerPosteriorSampler

from timeit import default_timer as timer
from datetime import timedelta

import matplotlib.pyplot as plt

# %% Avoid issuing memory warning due to number of plots

plt.rcParams.update({'figure.max_open_warning': 0})

# %% Set up output directory

OUTDIR = os.path.join("output", "pp_mala", "multiple_chains", "normal_prior", "var003", "drift0p02")

if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

# %% Load IRIS data

iris = Iris()
dataloader = DataLoader(iris, batch_size=150)

# %% Setup MLP model

# hparams = mlp.Hyperparameters(dims=[4, 3, 3])
hparams = mlp.Hyperparameters(dims=[4, 3, 3], activations=[lambda x: 0.01 * x, torch.sigmoid])
model = mlp.MLP(hparams=hparams, dtype=torch.float64)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(3)*torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Set number of chains, iterations, burnin iterations and MALA drift step

num_chains = 5
num_iterations = 1100
num_burnin = 100
num_post_burnin = num_iterations - num_burnin
num_powers = 10

drift_step = 0.2
per_chain_samplers = num_powers * [['MALA', {'step': drift_step}]]

msg = "Run {:" + str(len(str(num_chains))) + "}, duration {}"

# %% Run simulation and save output

for i in [0]: # range(num_chains):
    # Setup PowerPosteriorSampler
    theta0 = model.prior.sample()
    sampler = PowerPosteriorSampler(model, theta0, dataloader, per_chain_samplers)
    
    # Run MALA sampler
    start_time = timer()
    sampler.run(num_iterations=num_iterations, num_burnin=num_burnin)
    end_time = timer()
    print(msg.format(i+1, timedelta(seconds=end_time-start_time)))
    
    # Store all parameters in a single torch tensor
    chain = torch.empty(num_post_burnin, model.num_params())
    for j in range(model.num_params()):
        chain[:, j] = torch.tensor(sampler.get_chain().get_theta(j))

    ## Plot traces of simulated chain
    for i in range(model.num_params()):
        plt.figure()
        plt.plot(sampler.get_chain().get_theta(i))
