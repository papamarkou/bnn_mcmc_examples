# %% Import packages

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from datetime import timedelta
from timeit import default_timer as timer

from bnn_mcmc_examples.datasets import load_xydataset_from_file
from bnn_mcmc_examples.datasets.noisy_xor.data1.constants import training_data_path
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.constants import dtype
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.model import model

# %% Load dataloader

_, dataloader = load_xydataset_from_file(training_data_path, dtype=dtype)
data, labels = next(iter(dataloader))

# %%

def train(model, loss_fn, dataloader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for batch_idx, (input, target) in enumerate(dataloader):
            def closure():
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                return loss

            optimizer.step(closure)

# %% Setup optimizer

lr = 0.001
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# %% Define loss function

loss_function = nn.BCELoss(reduction='sum')

# %% Set number of batches

num_batches = 1000 # 2000

# %% Initialize torch tensor that will hold the loss values

loss_vals = torch.empty(num_batches, dtype=dtype, requires_grad=True)

# %% Initialize model parameters

theta0 = model.prior.sample()
model.set_params(theta0.clone().detach())

# %% Run optimizer

# loss_vals = torch.empty(num_batches, dtype=torch.float, requires_grad=True)

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
