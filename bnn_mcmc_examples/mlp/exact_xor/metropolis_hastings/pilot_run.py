# %% Import packages

import numpy as np
import torch

from datetime import timedelta
from timeit import default_timer as timer

from eeyore.chains import ChainList
from eeyore.kernels import NormalKernel
from eeyore.samplers import MetropolisHastings

from bnn_mcmc_examples.mlp.exact_xor.constants import num_burnin_epochs, num_epochs, verbose, verbose_step
from bnn_mcmc_examples.mlp.exact_xor.dataloader import dataloader
from bnn_mcmc_examples.mlp.exact_xor.metropolis_hastings.constants import sampler_output_pilot_path
from bnn_mcmc_examples.mlp.exact_xor.model import model

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

sampler.run(num_epochs=num_epochs, num_burnin_epochs=num_burnin_epochs, verbose=verbose, verbose_step=verbose_step)

end_time = timer()
print("Time taken: {}".format(timedelta(seconds=end_time-start_time)))

# %% Save chain array

sampler.get_chain().to_chainfile(keys=['sample', 'accepted'], path=sampler_output_pilot_path, mode='w')
