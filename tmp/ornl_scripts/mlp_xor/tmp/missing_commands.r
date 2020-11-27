print(paste("Mean acceptance: ", round(100*mean_acceptance, 2), sep=''))

print("Min, median and max ESS obtained via mcmc package:")
print(c(round(min(ess_mcmc), 0), round(median(ess_mcmc), 0), round(max(ess_mcmc), 0)))



print(max(diagnostics$Rhat))
print(max(rhats))
