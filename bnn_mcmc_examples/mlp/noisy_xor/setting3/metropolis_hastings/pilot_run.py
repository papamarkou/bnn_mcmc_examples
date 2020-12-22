# %% Import packages

import numpy as np
import torch

from datetime import timedelta
from timeit import default_timer as timer

from eeyore.chains import ChainList
from eeyore.kernels import NormalKernel
from eeyore.samplers import MetropolisHastings

from bnn_mcmc_examples.datasets import load_xydataset_from_file
from bnn_mcmc_examples.datasets.noisy_xor.data2.constants import training_data_path
from bnn_mcmc_examples.mlp.noisy_xor.setting3.constants import (
    dtype, mcmc_batch_size, num_mcmc_burnin_epochs, num_mcmc_epochs, verbose, mcmc_verbose_step
)
from bnn_mcmc_examples.mlp.noisy_xor.setting3.metropolis_hastings.constants import sampler_output_pilot_path
from bnn_mcmc_examples.mlp.noisy_xor.setting3.model import model

# %% Load dataloader

_, dataloader = load_xydataset_from_file(training_data_path, dtype=dtype, batch_size=mcmc_batch_size)

# %% Setup proposal variance and proposal kernel for Metropolis-Hastings sampler

proposal_scale = np.sqrt(0.02)

kernel = NormalKernel(
    torch.zeros(model.num_params(), dtype=model.dtype),
    torch.full([model.num_params()], proposal_scale, dtype=model.dtype)
)

# %% Setup Metropolis-Hastings sampler

sampler = MetropolisHastings(
    model, theta0=model.prior.sample(), dataloader=dataloader, kernel=kernel, chain=ChainList(keys=['sample', 'accepted'])
)

# %% Run Metropolis-Hastings sampler

start_time = timer()

sampler.run(
    num_epochs=num_mcmc_epochs, num_burnin_epochs=num_mcmc_burnin_epochs, verbose=verbose, verbose_step=mcmc_verbose_step
)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Save chain array

sampler.get_chain().to_chainfile(keys=['sample', 'accepted'], path=sampler_output_pilot_path, mode='w')
