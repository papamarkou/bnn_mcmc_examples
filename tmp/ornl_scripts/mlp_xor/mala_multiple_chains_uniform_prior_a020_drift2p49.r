library(stringr)
library(coda)
library(mcmc)

OUTDIR <- file.path("output", "mala", "multiple_chains", "uniform_prior", "a020", "drift2p49")

source(file.path("..", "metrics.r"))

num_chains <- 10
num_post_burnin <- 100000
num_pars <- 9

mcchains <- vector(mode="list", length=num_chains)
for (i in seq(1, num_chains)) {
  print(paste("Loading chain", str_pad(i, 2, pad='0')))
  mcchains[[i]] <- mcmc(
    read.table(file.path(OUTDIR, paste('chain', str_pad(i, 2, pad='0'), '.csv', sep='')), header=FALSE, sep=",")
  )
}
mcchains <- mcmc.list(mcchains)

means <- rep(0, num_pars)
for (i in seq(1, num_chains)) {
  means <- means+apply(mcchains[[i]], 2, mean)
}
means <- means/num_chains

print("Mean:")
print(round(means, 4))

acceptance <- vector(mode="numeric", length=num_chains)
for (i in seq(1, num_chains)) {
  accepted <- read.table(file.path(OUTDIR, paste('accepted', str_pad(i, 2, pad='0'), '.txt', sep='')), header=FALSE)
  acceptance[i] <- mean(accepted$V1)
}

mean_acceptance <- mean(acceptance)

print(paste("Mean acceptance: ", round(100*mean_acceptance, 2), sep=''))

gelman_diagnostic <- gelman.diag(mcchains)

print(paste("Multivariate PSRF: ", gelman_diagnostic$mpsrf, sep=''))

mcchain_crosscorr <- crosscorr(mcchains)

print("Empirical covariance:")
print(mcchain_crosscorr)

ess_conda <- rep(0, times=num_pars)
for (i in seq(1, num_chains)) {
  # print(paste("ESS for simulation ", i, ":", sep=''))
  # print(effectiveSize(mcchains[[i]]))
  ess_conda <- ess_conda+effectiveSize(mcchains[[i]])
}
ess_conda <- ess_conda/num_chains

print("Mean ESS obtained via conda package:")
print(ess_conda)

ess_mcmc <- rep(0, times=num_pars)
for (i in seq(1, num_chains)) {
  ess_mcmc <- ess_mcmc+apply(as.matrix(mcchains[[i]]), 2, ess)
}
ess_mcmc <- ess_mcmc/num_chains

print("Mean ESS obtained via mcmc package:")
print(ess_mcmc)

print("Min, median and max ESS obtained via mcmc package:")
print(c(round(min(ess_mcmc), 0), round(median(ess_mcmc), 0), round(max(ess_mcmc), 0)))

for (i in seq(1, num_chains)) {
  for (j in seq(1, num_pars)) {
    chain <- as.vector(mcchains[[i]][, j])

    plot(chain, type='l', main=paste("Simulation ", i, ", parameter ", j, sep=''))
    acf(chain, lag.max=100, main=paste("Simulation ", i, ", parameter ", j, sep=''))
    hist(chain, breaks=20, main=paste("Simulation ", i, ", parameter ", j, sep=''))
  }
}
