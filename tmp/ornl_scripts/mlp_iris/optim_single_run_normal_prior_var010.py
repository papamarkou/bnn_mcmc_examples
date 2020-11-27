# %% Import packages

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

import matplotlib.pyplot as plt

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

hparams = mlp.Hyperparameters(dims=[4, 3, 3], activations=[torch.sigmoid, None])
model = mlp.MLP(hparams=hparams, loss=lambda x, y: nn.CrossEntropyLoss(reduction='sum')(x, torch.argmax(y, 1)), dtype=torch.float)
model.prior = Normal(
    torch.zeros(model.num_params(), dtype=model.dtype),
    np.sqrt(10)*torch.ones(model.num_params(), dtype=model.dtype)
)

# %% Setup optimizer

lr = 0.001
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# %% Set number of batches

num_batches = 5000

# %% Initialize torch tensors and lists that will hold the output

losses = torch.empty(num_batches, dtype=torch.float, requires_grad=True)

# %% Critierion for accepting optimization

def accept_optimization(logit):
    if torch.all(torch.argmax(logit.softmax(1), 1) == torch.argmax(test_labels, 1)):
        return False
    else:
        return True

# %% Run simulation

# Initialize model parameters
theta0 = model.prior.sample()
model.set_params(theta0.clone().detach())

# Run optimizer
start_time = timer()

for i in range(num_batches):
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

    losses[i] = loss_val.clone().detach().item()


end_time = timer()
runtime = timedelta(seconds=end_time - start_time)
print("Duration: {}".format(runtime))

# %% Plot loss

plt.plot(losses.detach().numpy())
