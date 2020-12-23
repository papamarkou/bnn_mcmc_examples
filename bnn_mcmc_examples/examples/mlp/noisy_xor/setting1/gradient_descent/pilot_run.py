# %% Import packages

from torch.optim import SGD

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.dataloader import dataloader
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.model import model
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.gradient_descent.optimizer import loss_fn, lr, momentum
from bnn_mcmc_examples.optim import train

# %% Setup SGD optimizer

optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

# %% Train model

loss_vals = train(model, dataloader, optimizer, num_epochs, loss_fn=loss_fn, save_loss=True)
