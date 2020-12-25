# %% Import packages

import matplotlib.pyplot as plt

# %% Plot training loss

plt.plot(summaries['loss_vals'])

# %% Plot training accuracy

plt.plot(summaries['metric_vals'])

# %% Plot running mean of training accuracy

plt.plot(summaries['metric_mean_vals'])
