# %% Import packages

import pandas as pd

from sklearn.model_selection import train_test_split

from bnn_mcmc_examples.datasets import data_paths

# %% Load Pima data

x = pd.read_csv(data_paths['pima'].joinpath('x.csv'))
y = pd.read_csv(data_paths['pima'].joinpath('y.csv'))

# %% Split data to training and test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=100, stratify=y)
