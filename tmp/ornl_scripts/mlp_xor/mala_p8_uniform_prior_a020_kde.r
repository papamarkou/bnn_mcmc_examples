library(bde) # For kernel density estimation for density of bounded support

OUTDIR <- file.path("output", "mala", "multiple_chains", "uniform_prior", "a020", "drift2p49")

store <- TRUE

nmcmc <- 100000
startmcmc <- 50000

i <- 10 # Chain
j <- 8 # Coordinate

chain <- read.table(file.path(OUTDIR, paste('chain', str_pad(i, 2, pad='0'), '.csv', sep='')), header=FALSE, sep=",")

chain_kde <- chen99Kernel(chain[, j], lower.limit=-20, upper.limit=20, b=0.01)

x_prior <- seq(-20, 20, by=0.01)
y_prior <- dunif(x_prior, min=-20, max=20)

chainmean <- mean(chain[, j])

cols <- c("red", "blue")

if (store) {
  pdf(file=file.path(OUTDIR, paste("mala_p", j,"_uniform_prior_a020_kde.pdf", sep='')), width=10, height=6)
}

oldpar <- par(no.readonly=TRUE)

par(fig=c(0, 1, 0, 1), mar=c(2.25, 4, 3.5, 1)+0.1)

plot(
  chain_kde,
  type="l",
  xlim=c(-20, 20),
  ylim=c(0, 0.041),
  col=cols[1],
  xlab="Iteration",
  ylab="",
  main=bquote(paste(theta[.(j)], ": U(-20, 20) prior", sep=" ")),
  cex.axis=1.8,
  cex.main=2,
  cex.lab=2,
  lwd=2,
  xaxt="n",
  yaxt="n"
)

axis(
  1,
  at=seq(-20, 20, by=5),
  labels=seq(-20, 20, by=5),
  cex.axis=1.8,
  las=1
)

axis(
  2,
  at=seq(0, 0.04, by=0.01),
  labels=seq(0, 0.04, by=0.01),
  cex.axis=1.8,
  las=1
)

lines(
  x_prior,
  y_prior,
  type="l",
  col=cols[2],
  lwd=2
)

legend(
  "topright",
  c("Posterior", "Prior"),
  lty=c(1, 1),
  lwd=c(5, 5),
  col=cols,
  cex=1.8,
  bty="n"
)

par(oldpar)

if (store) {
  dev.off()
}
