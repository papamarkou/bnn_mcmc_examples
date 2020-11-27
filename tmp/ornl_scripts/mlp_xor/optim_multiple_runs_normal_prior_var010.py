# %% Import packages

import os
import csv

import numpy as np

import torch
# import torch.nn.functional as F
# import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.distributions import Normal

from eeyore.data import XOR
from eeyore.models import mlp

from timeit import default_timer as timer
from datetime import timedelta

# %% Set up output directory

OUTDIR = os.path.join("output", "optim", "multiple_runs", "normal_prior", "var010", "iters010000")

if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

# %% Load XOR data

num_samples = 4

xor = XOR(dtype=torch.float)
dataloader = DataLoader(xor, batch_size=num_samples)

data, labels = next(iter(dataloader))

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=[2, 2, 1], bias=[True, True], activations=[torch.sigmoid, torch.sigmoid])
model = mlp.MLP(hparams=hparams, dtype=torch.float)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(10)*torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Setup optimizer

lr = 1.
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# %% Define loss function

loss_function = nn.BCELoss(reduction='sum')

# Same as
# loss_function = lambda logit, labels: F.binary_cross_entropy(logit, labels, reduction='sum')
#
# In other words, the following two snippets give the same output
# logit = model(data)
# loss = nn.BCELoss(reduction='sum')(logit, labels)
#
# logit = model(data)
# loss = F.binary_cross_entropy(logit, labels, reduction='sum')
#
# Same as minus the log-likelihood of the MLP in eeyore

# %% Set number of optimizations and of batches

num_optimizations = 10000
num_batches = 2000

# %% Initialize torch tensors and lists that will hold the output

thetas = torch.empty(num_optimizations, model.num_params(), dtype=torch.float)
logits = torch.empty(num_optimizations, num_samples, dtype=torch.float)
losses = torch.empty(num_optimizations, dtype=torch.float, requires_grad=True)

runtimes = []

# %%

def accept_optimization(logit, thres):
    if (torch.any(torch.isnan(logit)) or (not (
        ((logit[0] < 1 - thres) and (logit[1] > thres) and (logit[2] > thres) and (logit[3] < 1 - thres))
    ))):
        return False
    else:
        return True

# %% Run simulation

i = 0 # Count of optimizations with correct classification on training set
j = 0 # Count of optimizations with wrong classification on training set

while (i < num_optimizations):
    # Initialize model parameters
    theta0 = model.prior.sample()
    model.set_params(theta0.clone().detach())

    # Run optimizer
    msg = "Simulation {}: duration {}, {:" + str(len(str(num_optimizations))) + "} accepted"
    start_time = timer()

    for _ in range(num_batches):
        # optimizer.zero_grad()

        logit = model(data)

        loss = loss_function(logit, labels)

        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                p.data -= lr * p.grad
                p.grad.zero_()
        # optimizer.step()

    if accept_optimization(logit, 0.9):
        thetas[i, :] = torch.cat([p.clone().detach().view(-1) for p in model.parameters()])
        logits[i, :] = logit.clone().detach().reshape(num_samples)
        losses[i] = loss.clone().detach().item()

        i = i + 1
        acceptance = "accepted"
    else:
        j = j + 1
        acceptance = "rejected"

    end_time = timer()
    runtime = timedelta(seconds=end_time - start_time)
    runtimes.append(runtime)
    print(msg.format(acceptance, runtime, i))

# %% Store the ouput in file

# Save optimization solutions in file
with open(os.path.join(OUTDIR, "thetas.csv"), 'w') as file:
    np.savetxt(file, thetas.cpu().detach().numpy(), delimiter=',', newline='\n', header='')

# Save logits in file
with open(os.path.join(OUTDIR, "logits.csv"), 'w') as file:
    np.savetxt(file, logits.cpu().detach().numpy(), delimiter=',', newline='\n', header='')

# Save losses in file
with open(os.path.join(OUTDIR, "losses.txt"), 'w') as file:
    np.savetxt(file, losses.cpu().detach().numpy(), newline='\n', header='')

# Save runtimes in file
with open(os.path.join(OUTDIR, "runtimes.txt"), 'w') as file:
    writer = csv.writer(file)
    for t in runtimes:
        writer.writerow([t])

# Save number of failed optimizations in file
with open(os.path.join(OUTDIR, "num_failures.txt"), 'w') as file:
    writer = csv.writer(file)
    writer.writerow([j])
