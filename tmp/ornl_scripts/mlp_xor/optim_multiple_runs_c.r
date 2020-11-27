OUTDIR00 <- file.path("output", "optim", "multiple_runs")
OUTDIR01 <- file.path("output", "mala", "multiple_chains", "uniform_prior", "a020", "drift2p49")

store <- TRUE

nmcmc <- 100000
startmcmc <- 50000

i <- 3 # Chain: 3, 7, 8, 9, 10
j <- 4 # Coordinate

chain <- read.table(file.path(OUTDIR01, paste('chain', str_pad(i, 2, pad='0'), '.csv', sep='')), header=FALSE, sep=",")

chainmean <- mean(chain[, j])

# mode01 <- c(-10.839866, -4.586011)
# mode02 <- c(4.322782, 10.871630)

if (store) {
  pdf(file=file.path(OUTDIR00, "optim_multiple_runs_c.pdf"), width=10, height=6)
}

oldpar <- par(no.readonly=TRUE)

par(fig=c(0, 1, 0, 1), mar=c(2.25, 4, 3.5, 1)+0.1)

plot(
  startmcmc:nmcmc,
  chain[startmcmc:nmcmc, j],
  type="l",
  ylim=c(-20, 20),
  col="steelblue2",
  xlab="Iteration",
  ylab="",
  main=bquote(paste(theta[.(j)], ": U(-20,20) prior, MALA", sep=" ")),
  cex.axis=1.8,
  cex.main=2,
  cex.lab=2,
  yaxt="n"
)

axis(
  2,
  at=seq(-20, 20, by=5),
  labels=seq(-20, 20, by=5),
  cex.axis=1.8,
  las=1
)

lines(
  1:nmcmc,
  rep(chainmean, nmcmc),
  type="l",
  col="black",
  lwd=2
)

par(oldpar)

if (store) {
  dev.off()
}
