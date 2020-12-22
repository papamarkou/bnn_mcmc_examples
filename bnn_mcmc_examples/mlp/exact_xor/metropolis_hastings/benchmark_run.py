# %% Import packages

import numpy as np
import torch

from eeyore.kernels import NormalKernel
from eeyore.samplers import MetropolisHastings

from bnn_mcmc_examples.mlp.exact_xor.constants import (
    num_chains, num_mcmc_epochs, num_mcmc_burnin_epochs, verbose, mcmc_verbose_step
)
from bnn_mcmc_examples.mlp.exact_xor.dataloader import dataloader
from bnn_mcmc_examples.mlp.exact_xor.metropolis_hastings.constants import sampler_output_path
from bnn_mcmc_examples.mlp.exact_xor.model import model

# %% Setup proposal variance and proposal kernel for Metropolis-Hastings sampler

proposal_scale = np.sqrt(3.5)

kernel = NormalKernel(
    torch.zeros(model.num_params(), dtype=model.dtype),
    torch.full([model.num_params()], proposal_scale, dtype=model.dtype)
)

# %% Setup Metropolis-Hastings sampler

sampler = MetropolisHastings(model, theta0=model.prior.sample(), dataloader=dataloader, kernel=kernel)

# %% Benchmark Metropolis-Hastings sampler

sampler.benchmark(
    num_chains=num_chains,
    num_epochs=num_mcmc_epochs,
    num_burnin_epochs=num_mcmc_burnin_epochs,
    path=sampler_output_path,
    check_conditions=lambda chain, runtime : 0.15 <= chain.acceptance_rate() <= 0.50,
    verbose=verbose,
    verbose_step=mcmc_verbose_step,
    print_acceptance=True,
    print_runtime=True
)
