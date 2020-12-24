# %% Import packages

from sklearn.metrics import accuracy_score

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.dataloaders import test_dataloader, training_dataloader

# %% Load training data and labels

training_data, training_labels = next(iter(training_dataloader))

# %% Compute training logits

training_logits = model(training_data)

# Make training predictions

training_preds = training_logits.squeeze() > 0.5

# %% Compute training accuracy

print(accuracy_score(training_preds, training_labels.squeeze()))
# (preds == labels.squeeze()).sum()/len(labels.squeeze())

# %% Show parameter values

print(model.get_params())

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
