library(stringr)
library(rstan)

OUTDIR <- file.path("output", "mala", "multiple_chains", "normal_prior", "var003")

source(file.path("..", "metrics.r"))

num_chains <- 4
num_post_burnin <- 100000
num_pars <- 27

mcchains <- array(data=NA, dim=c(num_post_burnin, num_chains, num_pars))
for (i in seq(1, num_chains)) {
  print(paste("Loading chain", str_pad(i, 2, pad='0')))
  mcchains[, i, ] <- as.matrix(read.table(
    file.path(OUTDIR, paste('chain', str_pad(i, 2, pad='0'), '.csv', sep='')), header=FALSE, sep=","
  ))
}

acceptance <- vector(mode="numeric", length=num_chains)
for (i in seq(1, num_chains)) {
  accepted <- read.table(file.path(OUTDIR, paste('accepted', str_pad(i, 2, pad='0'), '.txt', sep='')), header=FALSE)
  acceptance[i] <- mean(accepted$V1)
}

mean_acceptance <- mean(acceptance)

print(paste("Mean acceptance: ", round(100*mean_acceptance, 2), sep=''))

diagnostics <- monitor(mcchains)

diagnostics

rhats <- vector(mode="numeric", length=num_pars)
for (i in seq(1, num_pars)) {
  rhats[i] <- Rhat(mcchains[, , i])
}

bulk_ess <- vector(mode="numeric", length=num_pars)
for (i in seq(1, num_pars)) {
  bulk_ess[i] <- ess_bulk(mcchains[, , i])
}

tail_ess <- vector(mode="numeric", length=num_pars)
for (i in seq(1, num_pars)) {
  tail_ess[i] <- ess_tail(mcchains[, , i])
}

print(paste("Split R-hat:"))
print(diagnostics$Rhat)
print(max(diagnostics$Rhat))

print(paste("New R-hat:"))
print(rhats)
print(max(rhats))

print(paste("Bulk ESS from diagnostics:"))
print(diagnostics$Bulk_ESS)

print(paste("New bulk ESS:"))
print(bulk_ess)

print(paste("Tail ESS from diagnostics:"))
print(diagnostics$Tail_ESS)

print(paste("New tail ESS:"))
print(tail_ess)

for (i in seq(1, num_pars)) {
  for (j in seq(1, num_chains)) {
    chain <- as.vector(mcchains[, j, i])
    
    plot(chain, type='l', main=paste("Parameter ", i, ", simulation ", j, sep=''))
    acf(chain, lag.max=100, main=paste("Parameter ", i, ", simulation ", j, sep=''))
    hist(chain, breaks=20, main=paste("Parameter ", i, ", simulation ", j, sep=''))
  }
}
