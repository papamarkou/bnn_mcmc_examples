# %% Import packages

import torch

import kanga.plots as ps

from datetime import timedelta
from timeit import default_timer as timer

from eeyore.kernels import NormalKernel
from eeyore.samplers import MetropolisHastings

from bnn_mcmc_examples.mlp.noisy_xor.constants import num_burnin_epochs, num_epochs, verbose, verbose_step
from bnn_mcmc_examples.mlp.noisy_xor.data import dataloader
from bnn_mcmc_examples.mlp.noisy_xor.model import model

# %% Setup proposal variance and proposal kernel for Metropolis-Hastings sampler

proposal_scale = 0.1

kernel = NormalKernel(
    torch.zeros(model.num_params(), dtype=model.dtype),
    torch.full([model.num_params()], proposal_scale, dtype=model.dtype)
)

# %% Setup Metropolis-Hastings sampler

sampler = MetropolisHastings(model, theta0=model.prior.sample(), dataloader=dataloader, kernel=kernel)

# %% Run Metropolis-Hastings sampler

start_time = timer()

sampler.run(num_epochs=num_epochs, num_burnin_epochs=num_burnin_epochs, verbose=verbose, verbose_step=verbose_step)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Generate kanga ChainArray from eeyore ChainList

chain_array = sampler.get_chain().to_kanga()

# %% Compute acceptance rate

print('Acceptance rate: {}'.format(chain_array.acceptance_rate()))

# %% Compute Monte Carlo mean

print('Monte Carlo mean: {}'.format(chain_array.mean()))

# %% Compute Monte Carlo covariance

mc_cov_mat = chain_array.mc_cov()

# %% Compute Monte Carlo standard error

print('Monte Carlo standard error: {}'.format(chain_array.mc_se(mc_cov_mat=mc_cov_mat)))

# %% Compute multivariate ESS

print('Multivariate ESS: {}'.format(chain_array.multi_ess(mc_cov_mat=mc_cov_mat)))

# %% Plot traces of simulated chain

for i in range(model.num_params()):
    ps.trace(
        chain_array.get_param(i),
        title=r'Traceplot of $\theta_{{{}}}$'.format(i+1),
        xlabel='Iteration',
        ylabel='Parameter value'
    )

# %% Plot running means of simulated chain

for i in range(model.num_params()):
    ps.running_mean(
        chain_array.get_param(i),
        title=r'Running mean plot of parameter $\theta_{{{}}}$'.format(i+1),
        xlabel='Iteration',
        ylabel='Running mean'
    )

# %% Plot histograms of marginals of simulated chain

for i in range(model.num_params()):
    ps.hist(
        chain_array.get_param(i),
        bins=30,
        density=True,
        title=r'Histogram of parameter $\theta_{{{}}}$'.format(i+1),
        xlabel='Parameter value',
        ylabel='Parameter relative frequency'
    )
