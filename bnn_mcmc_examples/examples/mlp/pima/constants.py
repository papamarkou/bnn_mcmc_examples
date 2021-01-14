# %% Import packages

import numpy as np
import torch

from pathlib import Path

# %% Define constants

output_path = Path.home().joinpath('output', 'bnn_mcmc_examples', 'mlp', 'pima')

num_features = 4

mlp_dims = [num_features, 2, 1]

dtype = torch.float32

num_chains = 2 # 10

num_epochs = 11000 # 110000
num_burnin_epochs = 0
diagnostic_iter_thres = 1000 # 10000

verbose = True
verbose_step = 1000

pred_interval_x_l = -0.5
pred_interval_x_u = 1.5
pred_interval_num = 11 # 22
pred_interval_step = (pred_interval_x_u - pred_interval_x_l) / pred_interval_num

pred_interval_x1 = np.linspace(
    pred_interval_x_l + 0.5 * pred_interval_step,
    pred_interval_x_u - 0.5 * pred_interval_step,
    num=pred_interval_num
)
pred_interval_x2 = np.linspace(
    pred_interval_x_l + 0.5 * pred_interval_step,
    pred_interval_x_u - 0.5 * pred_interval_step,
    num=pred_interval_num
)

pred_iter_thres = 1000 # 100000
