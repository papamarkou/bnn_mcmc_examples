# %% Import packages

from sklearn.metrics import accuracy_score

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.optim.dataloaders import test_dataloader

# %% Load test data and labels

test_data, test_labels = next(iter(test_dataloader))

# %% Compute test logits

test_logits = model(test_data)

# Make test predictions

test_preds = test_logits.squeeze() > 0.5

# %% Compute test accuracy

print(accuracy_score(test_preds, test_labels.squeeze()))
# (preds == labels.squeeze()).sum()/len(labels.squeeze())

# %% Show parameter values

print(model.get_params())
