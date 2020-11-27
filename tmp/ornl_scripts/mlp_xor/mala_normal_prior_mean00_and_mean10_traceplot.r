OUTDIR01 <- file.path("output", "mala", "multiple_chains", "normal_prior", "var003")
OUTDIR02 <- file.path("output", "mala", "multiple_chains", "normal_prior", "mean10")

store <- TRUE

nmcmc <- 100000
startmcmc <- 50000

i <- 1 # Chain
j <- 4 # Coordinate

chain01 <- read.table(
    file.path(OUTDIR01, paste('chain', str_pad(i, 2, pad='0'), '.csv', sep='')), header=FALSE, sep=","
)
chain02 <- read.table(
  file.path(OUTDIR02, paste('chain', str_pad(i, 2, pad='0'), '.csv', sep='')), header=FALSE, sep=","
)

chainmean01 <- mean(chain01[, j])
chainmean02 <- mean(chain02[, j])

if (store) {
  pdf(file=file.path(OUTDIR01, "mala_normal_prior_mean00_and_mean10_traceplot.pdf"), width=10, height=6)
}

oldpar <- par(no.readonly=TRUE)

par(fig=c(0, 1, 0, 1), mar=c(2.25, 4, 3.5, 1)+0.1)

cols <- c("steelblue2", "darkgrey")

plot(
  startmcmc:nmcmc,
  chain01[startmcmc:nmcmc, j],
  type="l",
  ylim=c(-34, 34),
  col=cols[1],
  xlab="Iteration",
  ylab="",
  main=bquote(paste(theta[.(j)], ": N(0,3I) and N(10,3I) priors, MALA", sep=" ")),
  cex.axis=1.8,
  cex.main=2,
  cex.lab=2,
  yaxt="n"
)

lines(
  startmcmc:nmcmc,
  chain02[startmcmc:nmcmc, j],
  type="l",
  ylim=c(-34, 34),
  col=cols[2]
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
  rep(chainmean01, nmcmc),
  type="l",
  col="black",
  lwd=2
)

lines(
  1:nmcmc,
  rep(chainmean02, nmcmc),
  type="l",
  col="black",
  lwd=2
)

legend(
  "topright",
  c("N(10, 3I)", "N(0, 3I)"),
  lty=c(1, 1),
  lwd=c(5, 5),
  col=c(cols[2], cols[1]),
  cex=1.8,
  bty="n"
)

par(oldpar)

if (store) {
  dev.off()
}
