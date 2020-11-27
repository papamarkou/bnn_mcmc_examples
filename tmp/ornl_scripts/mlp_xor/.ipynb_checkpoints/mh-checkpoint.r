library(stringr)
library(mcmc)

OUTDIR <- "output"

source("../metrics.r")

ess <- function(chain, mcvar) {
  return(var(chain)/mcvar)
}

npars <- 9

accepted <- read.table(file.path(OUTDIR, 'accepted.txt'), header=FALSE)

mean(accepted$V1)

means <- vector(mode="numeric", length=npars)
mcvar <- vector(mode="numeric", length=npars)
mcess <- vector(mode="numeric", length=npars)

for (i in seq(1, npars)) {
  print(i)
  chain <- read.table(file.path(OUTDIR, paste('mh_chain', str_pad(i, 2, pad='0'), '.txt', sep='')), header=FALSE)

  means[i] <- mean(chain$V1)

  iseq <- initseq(chain$V1)
  mcvar[i] <- iseq$var.con/length(chain$V1)

  mcess[i] <- ess(chain$V1, mcvar[i])
}

summaries <- data.frame(means=means, mcvar=mcvar, mcess=mcess)

plot(chain$V1, type='l')

acf(chain$V1, lag.max=100)

hist(chain$V1, breaks=20)
plot(density(chain$V1))
