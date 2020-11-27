# %% Import packages

import torch

from eeyore.kernels import NormalKernel
from eeyore.samplers import MetropolisHastings

from bnn_mcmc_examples.mlp.noisy_xor.constants import (
    num_chains, num_epochs, num_burnin_epochs, verbose, verbose_step
)
from bnn_mcmc_examples.mlp.noisy_xor.data import dataloader
from bnn_mcmc_examples.mlp.noisy_xor.metropolis_hastings.constants import sampler_output_path
from bnn_mcmc_examples.mlp.noisy_xor.model import model

# %% Setup proposal variance and proposal kernel for Metropolis-Hastings sampler

proposal_scale = 1.55

kernel = NormalKernel(
    torch.zeros(model.num_params(), dtype=model.dtype),
    torch.full([model.num_params()], proposal_scale, dtype=model.dtype)
)

# %% Setup Metropolis-Hastings sampler

sampler = MetropolisHastings(model, theta0=model.prior.sample(), dataloader=dataloader, kernel=kernel)

# %% Benchmark Metropolis-Hastings sampler

sampler.benchmark(
    num_chains=num_chains,
    num_epochs=num_epochs,
    num_burnin_epochs=num_burnin_epochs,
    path=sampler_output_path,
    check_conditions=None,
    verbose=verbose,
    verbose_step=verbose_step,
    print_acceptance=True,
    print_runtime=True
)
