library(stringr)

OUTDIR <- "output"

store <- TRUE

SAMPLERDIRS <- c(
  file.path("output", "mala", "multiple_chains", "normal_prior", "mean10"),
  file.path("output", "mala", "multiple_chains", "normal_prior", "var003"),
  file.path("output", "mala", "multiple_chains", "normal_prior", "var010"),
  file.path("output", "mh", "multiple_chains", "normal_prior", "var003"),
  file.path("output", "mala", "multiple_chains", "normal_prior", "var100"),
  file.path("output", "mala", "multiple_chains", "uniform_prior", "a020", "drift2p49")
)

nsamplerdirs <- length(SAMPLERDIRS)

nmcmc <- 100000

chain_id <- c(1, 1, 1, 1, 1, 3) # Chain
j <- 4 # Coordinate

maxlag <- 40

cors <- matrix(data=NA, nrow=maxlag+1, ncol=nsamplerdirs)

for (k in 1:nsamplerdirs) {
  chain <- read.table(
    file.path(SAMPLERDIRS[k], paste('chain', str_pad(chain_id[k], 2, pad='0'), '.csv', sep='')),
    header=FALSE,
    sep=","
  )
  cors[, k] <- acf(chain[, j], lag.max=maxlag, demean=TRUE, plot=FALSE)$acf
}

sqrtnmcmc <- sqrt(nmcmc)

cols <- c("green", "blue", "purple", "brown", "orange", "red")

if (store) {
  pdf(file=file.path(OUTDIR, "acf_plot.pdf"), width=10, height=6)
}

oldpar <- par(no.readonly=TRUE)

par(fig=c(0, 1, 0, 1), mar=c(2.25, 4, 3.5, 1)+0.1)

plot(
  0:maxlag,
  cors[, 1],
  type="o",
  ylim=c(-0.2, 1.),
  col=cols[1],
  lwd=2,
  pch=20,
  xlab="",
  ylab="",
  cex.axis=1.8,
  cex.lab=2,
  yaxt="n"
)

axis(
  2,
  at=seq(-0.2, 1, by=0.2),
  labels=seq(-0.2, 1, by=0.2),
  cex.axis=1.8,
  las=1
)

for (k in 2:nsamplerdirs) {
  lines(
    0:maxlag,
    cors[, k],
    type="o",
    col=cols[k],
    lwd=2,
    pch=20
  )
}

abline(h=1.96/sqrtnmcmc, lty=2)
abline(h=-1.96/sqrtnmcmc, lty=2)

par(fig=c(0, 1, 0.89, 1), mar=c(0, 0, 0, 0), new=TRUE)

plot.new()

sampler_labels <- c(
  "MALA, N(10, 3I)",
  "MALA, N(0, 3I)",
  "MALA, N(0, 10I)",
  "MH, N(0, 3I)",
  "MALA, N(0, 100I)",
  "MALA, U(-20, 20)"
)

legend(
  "center",
  sampler_labels,
  lty=c(1, 1, 1, 1, 1, 1),
  lwd=c(5, 5, 5, 5, 5, 5),
  col=cols,
  cex=1.4,
  bty="n",
  text.width=0.25,
  ncol=3
)

par(oldpar)

if (store) {
  dev.off()
}
