library(stringr)
library(coda)
library(mcmc)

OUTDIR <- file.path("output", "mala", "single_chain")

source(file.path("..", "metrics.r"))

num_pars <- 19

chain <- read.table(file.path(OUTDIR, 'chain.csv'), header=FALSE, sep=',')

accepted <- read.table(file.path(OUTDIR, 'accepted.txt'), header=FALSE)

acceptance <- mean(accepted$V1)

print(paste("Acceptance: ", acceptance, sep=''))

mcmean <- apply(chain, 2, mean)

print("Monte Carlo mean:")
print(mcmean)

ess_conda <- effectiveSize(chain)

print("Mean ESS obtained via conda package:")
print(ess_conda)

ess_mcmc <- apply(chain, 2, ess)

print("Mean ESS obtained via mcmc package:")
print(ess_mcmc)

for (i in seq(1, num_pars)) {
  plot(chain[, i], type='l', main=paste("Parameter ", i, sep=''))
  acf(chain[, i], lag.max=100, main=paste("Parameter ", i, sep=''))
  hist(chain[, i], breaks=20, main=paste("Parameter ", i, sep=''))
}
