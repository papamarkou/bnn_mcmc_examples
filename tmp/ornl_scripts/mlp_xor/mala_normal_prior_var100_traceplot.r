OUTDIR <- file.path("output", "mala", "multiple_chains", "normal_prior", "var100")

store <- TRUE

nmcmc <- 100000
startmcmc <- 50000

i <- 1 # Chain
j <- 4 # Coordinate

chain <- read.table(file.path(OUTDIR, paste('chain', str_pad(i, 2, pad='0'), '.csv', sep='')), header=FALSE, sep=",")

chainmean <- mean(chain[, j])

if (store) {
  pdf(file=file.path(OUTDIR, "mala_normal_prior_var100_traceplot.pdf"), width=10, height=6)
}

oldpar <- par(no.readonly=TRUE)

par(fig=c(0, 1, 0, 1), mar=c(2.25, 4, 3.5, 1)+0.1)

plot(
  startmcmc:nmcmc,
  chain[startmcmc:nmcmc, j],
  type="l",
  ylim=c(-34, 34),
  col="steelblue2",
  xlab="Iteration",
  ylab="",
  main=bquote(paste(theta[.(j)], ": N(0,100I) prior, MALA", sep=" ")),
  cex.axis=1.8,
  cex.main=2,
  cex.lab=2,
  yaxt="n"
)

axis(
  2,
  at=seq(-35, 35, by=5),
  labels=seq(-35, 35, by=5),
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

