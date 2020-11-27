# %% Load packages

import matplotlib.pyplot as plt
import numpy as np

# %% Simulate noisy XOR points

n = 250
s = 0.55

x = np.vstack([
    np.column_stack([np.random.rand(n) - s, np.random.rand(n) - s]), # points corresponding to (0, 0)
    np.column_stack([np.random.rand(n) - s, np.random.rand(n) + s]), # points corresponding to (0, 1)
    np.column_stack([np.random.rand(n) + s, np.random.rand(n) - s]), # points corresponding to (1, 0)
    np.column_stack([np.random.rand(n) + s, np.random.rand(n) + s])  # points corresponding to (1, 1)
])

# %% Save simulated noisy XOR points

# %% Plot noisy XOR points

for i in range(4):
    plt.plot(x[i*n:(i+1)*n, 0], x[i*n:(i+1)*n, 1], 'o')
