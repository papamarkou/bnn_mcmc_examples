library(stringr)
library(corrplot)

store <- TRUE

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

mcchain_crosscorr <- crosscorr(mcchains)
colnames(mcchain_crosscorr) <- seq(1, 9)
rownames(mcchain_crosscorr) <- colnames(mcchain_crosscorr)

if (store) {
  pdf(file=file.path(OUTDIR, "mala_uniform_prior_a020_corrplot.pdf"), width=8.5, height=7)
}

# oldpar <- par(no.readonly=TRUE)

# par(fig=c(0, 1, 0, 1), mar=c(2.25, 4, 3.5, 1)+0.1)

corrplot(
  mcchain_crosscorr,
  type="upper",
  method="color",
  diag=FALSE,
  tl.col="black",
  tl.srt=45,
  addCoef.col="black",
  addgrid.col="black",
  mar=c(0, 0, 1, 0),
  cl.cex=1.8,
  tl.cex=1.8,
  number.cex=1.5,
  cl.length=11,
  cl.ratio=0.3
)

# par(oldpar)

if (store) {
  dev.off()
}

