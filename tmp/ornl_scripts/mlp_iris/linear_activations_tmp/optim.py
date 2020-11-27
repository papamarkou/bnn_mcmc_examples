# %% Import packages

import os
import csv

import numpy as np

import torch
# import torch.nn.functional as F
# import torch.optim as optim
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal

from eeyore.data import Iris
from eeyore.models import mlp

from timeit import default_timer as timer
from datetime import timedelta

# %% Set up output directory

OUTDIR = os.path.join("output", "optim", "multiple_runs", "normal_prior", "var010", "iters001000")

if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

# %%

class IrisTrainSet(Dataset):
    def __init__(self, iris):
        self.data = torch.cat((iris.data[0:40], iris.data[50:90], iris.data[100:140]), 0)
        self.labels = torch.cat((iris.labels[0:40], iris.labels[50:90], iris.labels[100:140]), 0)

    def __repr__(self):
        return f'Iris training dataset'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# %% Load iris data

num_samples = 150
num_train_samples = 120
num_test_samples = num_samples - num_train_samples

iris = Iris(dtype=torch.float)
iris_train = IrisTrainSet(iris)

dataloader = DataLoader(iris_train, batch_size=num_train_samples) # , shuffle=True)

data, labels = next(iter(dataloader))

test_data = torch.cat((iris.data[40:50], iris.data[90:100], iris.data[140:150]), 0)
test_labels = torch.cat((iris.labels[40:50], iris.labels[90:100], iris.labels[140:150]), 0)

# %% Setup MLP model

hparams = mlp.Hyperparameters(dims=[4, 3, 3], activations=[lambda x: 0.001 * x, None])
model = mlp.MLP(hparams=hparams, loss=lambda x, y: nn.CrossEntropyLoss(reduction='sum')(x, torch.argmax(y, 1)), dtype=torch.float)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(10)*torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Setup optimizer

lr = 0.001
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# %% Set number of optimizations and of batches

num_optimizations = 100
num_batches = 5000

# %% Initialize torch tensors and lists that will hold the output

thetas = torch.empty(num_optimizations, model.num_params(), dtype=torch.float)
# logits = torch.empty(num_optimizations, num_test_samples, 3, dtype=torch.float)
losses = torch.empty(num_optimizations, dtype=torch.float, requires_grad=True)

runtimes = []

# %% Critierion for accepting optimization

def accept_optimization(logit):
    if torch.all(torch.argmax(logit.softmax(1), 1) == torch.argmax(test_labels, 1)):
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
        
        data, labels = next(iter(dataloader))

        logit = model(data)

        loss_val = model.loss(logit, labels)

        loss_val.backward()

        with torch.no_grad():
            for p in model.parameters():
                p.data -= lr * p.grad
                p.grad.zero_()
        # optimizer.step()

    test_logit = model(test_data)

    if accept_optimization(test_logit):
        thetas[i, :] = torch.cat([p.clone().detach().view(-1) for p in model.parameters()])
        # logits[i, :, :] = test_logit.clone().detach()
        losses[i] = loss_val.clone().detach().item()

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
