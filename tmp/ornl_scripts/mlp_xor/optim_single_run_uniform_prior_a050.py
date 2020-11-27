# %% Import packages

import torch
# import torch.nn.functional as F
# import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.distributions import Uniform

from eeyore.data import XOR
from eeyore.models import mlp

import matplotlib.pyplot as plt

from timeit import default_timer as timer
from datetime import timedelta

# %% Load XOR data

xor = XOR(dtype=torch.float)
dataloader = DataLoader(xor, batch_size=4)

data, labels = next(iter(dataloader))
data, labels

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=[2, 2, 1], bias=[True, True], activations=[torch.sigmoid, torch.sigmoid])
model = mlp.MLP(hparams=hparams, dtype=torch.float)

a = 50.
model.prior = Uniform(
    -a * torch.ones(model.num_params(), dtype=model.dtype),
    +a * torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Setup optimizer

lr = 1.
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# %% Define loss function

loss_function = nn.BCELoss(reduction='sum')

# %% Set number of batches

num_batches = 2000

# %% Initialize torch tensor that will hold the loss values

loss_vals = torch.empty(num_batches, dtype=torch.float, requires_grad=True)

# %% Initialize model parameters

theta0 = model.prior.sample()
model.set_params(theta0.clone().detach())

# %% Run optimizer

loss_vals = torch.empty(num_batches, dtype=torch.float, requires_grad=True)

start_time = timer()

for i in range(num_batches):
    # optimizer.zero_grad()

    logit = model(data)

    loss = loss_function(logit, labels)

    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            p.data -= lr * p.grad
            p.grad.zero_()
    # optimizer.step()

    loss_vals[i] = loss.item()

end_time = timer()
runtime = timedelta(seconds=end_time - start_time)
print("Duration {}".format(timedelta(seconds=end_time - start_time)))

# %% Plot loss function

plt.plot(loss_vals.cpu().detach().numpy())

# %% Print model parameter values

for p in model.parameters():
    print(p.data)

# %% Print sigmoid output for training data

for d, l in zip(data, labels):
    print(d, l, model(d))

# model(data), data
