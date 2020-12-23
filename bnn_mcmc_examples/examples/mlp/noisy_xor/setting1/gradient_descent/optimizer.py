# %% Import packages

import torch.nn as nn

# %% Setup loss function

loss_fn = nn.BCELoss(reduction='sum')

# %% Setup optimizer hyper-parameters

lr = 0.001
momentum = 0.9
