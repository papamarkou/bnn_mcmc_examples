import torch

def predictive_prior(model, samples, x, y):
    integral_mean = 0
    n = 1

    for sample in samples:
        model.set_params(sample.clone().detach())
        lik_val = torch.exp(model.log_lik(x, y))

        if ((not torch.isinf(lik_val).item()) and (not torch.isnan(lik_val).item())):
            integral_mean = ((n - 1) * integral_mean + lik_val) / n
            n = n + 1

    return integral_mean
