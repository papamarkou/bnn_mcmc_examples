OUTDIR <- file.path("output", "mala", "multiple_chains", "normal_prior", "var003")

store <- TRUE

nmcmc <- 100000

i <- 1 # Chain
j <- 4 # Coordinate

chain <- read.table(file.path(OUTDIR, paste('chain', str_pad(i, 2, pad='0'), '.csv', sep='')), header=FALSE, sep=",")

chain_kde <- density(chain[, j])

x_prior <- seq(-7, 7, by=0.01)
y_prior <- dnorm(x_prior, mean=0, sd=sqrt(3))

chainmean <- mean(chain[, j])

cols <- c("red", "blue")

if (store) {
  pdf(file=file.path(OUTDIR, paste("mala_p", j,"_normal_prior_var003_kde.pdf", sep='')), width=10, height=6)
}

oldpar <- par(no.readonly=TRUE)

par(fig=c(0, 1, 0, 1), mar=c(2.25, 4, 3.5, 1)+0.1)

plot(
  chain_kde$x,
  chain_kde$y,
  type="l",
  xlim=c(-6, 6),
  ylim=c(0, 0.24),
  col=cols[1],
  xlab="Iteration",
  ylab="",
  main=bquote(paste(theta[.(j)], ": N(0,3I) prior", sep=" ")),
  cex.axis=1.8,
  cex.main=2,
  cex.lab=2,
  lwd=2,
  xaxt="n",
  yaxt="n"
)

axis(
  1,
  at=seq(-6, 6, by=2),
  labels=seq(-6, 6, by=2),
  cex.axis=1.8,
  las=1
)

axis(
  2,
  at=seq(0, 0.24, by=0.06),
  labels=seq(0, 0.24, by=0.06),
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
