# %% Import packages

from sklearn.metrics import accuracy_score

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.dataloaders import test_dataloader

# %% Load test data and labels

data, labels = next(iter(test_dataloader))

# %% Compute logits

logits = model(data)

# Make predictions

preds = logits.squeeze() > 0.5

# %% Compute accuracy

accuracy_score(preds, labels.squeeze())
# (preds == labels.squeeze()).sum()/len(labels.squeeze())
