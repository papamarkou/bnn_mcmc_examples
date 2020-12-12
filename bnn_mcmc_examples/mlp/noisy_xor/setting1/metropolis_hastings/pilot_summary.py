# %% Import packages

import kanga.plots as ps

from kanga.chains import ChainArray

from bnn_mcmc_examples.mlp.noisy_xor.setting1.metropolis_hastings.constants import sampler_output_pilot_path

# %% Load chain array

chain_array = ChainArray.from_file(keys=['sample', 'accepted'], path=sampler_output_pilot_path)

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
