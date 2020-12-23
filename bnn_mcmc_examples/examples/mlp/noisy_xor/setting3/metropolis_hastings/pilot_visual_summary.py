# %% Import packages

import kanga.plots as ps

from kanga.chains import ChainArray

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting3.metropolis_hastings.constants import sampler_output_pilot_path
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting3.model import model

# %% Load chain array

chain_array = ChainArray.from_file(keys=['sample', 'accepted'], path=sampler_output_pilot_path)

# %% Get the first 10000 iterations

# n = 10000
# chain_array.vals['sample'] = chain_array.vals['sample'][:n, ]
# chain_array.vals['accepted'] = chain_array.vals['accepted'][:n]

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