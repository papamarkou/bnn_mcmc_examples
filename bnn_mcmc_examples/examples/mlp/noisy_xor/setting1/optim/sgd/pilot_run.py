# %% Import packages

from sklearn.metrics import accuracy_score
# from torch.optim import Adam
from torch.optim import SGD

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.optim.constants import num_epochs
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.optim.dataloaders import training_dataloader
# from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.optim.sgd.optimizer import lr
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.optim.sgd.optimizer import lr, momentum
# from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.optim.sgd.optimizer import loss_fn, lr, momentum
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.model import model
from bnn_mcmc_examples.optim import train

# %% Setup SGD optimizer

model.set_params(model.prior.sample())

# optimizer = Adam(model.parameters(), lr=lr)
optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

# %% Train model

loss_vals, metric_vals, terminating_epoch = train(
    model,
    training_dataloader,
    optimizer,
    num_epochs,
    # loss_fn=loss_fn,
    save_loss=True,
    save_metric=True,
    terminate_early=False,
    pred_fn=lambda labels: labels.squeeze() > 0.5,
    metric_fn=lambda preds, labels: accuracy_score(preds, labels.squeeze()),
    stop_fn=lambda metric_val: metric_val >= 0.9
)
