# %% Import packages

from sklearn.metrics import accuracy_score
# from torch.optim import Adam
from torch.optim import SGD

from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.constants import num_optim_epochs
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.dataloaders import training_dataloader
# from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.gradient_descent.optimizer import lr
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.gradient_descent.optimizer import lr, momentum
# from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.gradient_descent.optimizer import loss_fn, lr, momentum
from bnn_mcmc_examples.examples.mlp.noisy_xor.setting1.model import model
from bnn_mcmc_examples.optim import train

# %% Setup SGD optimizer

model.set_params(model.prior.sample())

# optimizer = Adam(model.parameters(), lr=lr)
optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)

# %% Train model

loss_vals, metric_vals = train(
    model,
    training_dataloader,
    optimizer,
    num_optim_epochs,
    # loss_fn=loss_fn,
    save_loss=True,
    save_metric=True,
    terminate_early=True,
    pred_fn=lambda labels: labels.squeeze() > 0.5,
    metric_fn=lambda preds, labels: accuracy_score(preds, labels.squeeze()),
    stop_fn=lambda metric_val: metric_val >= 0.85
)
