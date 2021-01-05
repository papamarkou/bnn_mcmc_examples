# %% Load packages

import numpy as np

# %% Define constants

num_chains = 10

num_epochs = 110000
num_burnin_epochs = 0
diagnostic_iter_thres = 10000

verbose = True
verbose_step = 1000

ground_truth_x_l = -0.5
ground_truth_x_u = 1.5
ground_truth_num = 10
ground_truth_x1 = np.linspace(ground_truth_x_l, ground_truth_x_u, num=ground_truth_num)
ground_truth_x2 = np.linspace(ground_truth_x_l, ground_truth_x_u, num=ground_truth_num)
