# Examples of MCMC sampling for Bayesian neural networks

This repository provides examples for the manuscript
"Challenges in Bayesian inference via Markov chain Monte Carlo for neural networks".
The examples are organized in the form of a Python package,
called `bnn_mcmc_examples`.

For each MCMC simulation, the scripts are organized in the following order:
1. `pilot_run.py`
2. `pilot_numerical_summary.py`
3. `pilot_visual_summary.py`
4. `benchmark_run.py`
5. `benchmark_mc_cov.py`
6. `benchmark_numerical_summary.py`
7. `benchmark_visual_summary.py`
8. `benchmark_pred_posterior_vals_on_test.py`
9. `benchmark_preds_via_bm.py`
10. `benchmark_pred_accuracy_via_bm.py`

To summarize the comparison of samplers across MCMC simulations, run `mc_efficiency.py`.
