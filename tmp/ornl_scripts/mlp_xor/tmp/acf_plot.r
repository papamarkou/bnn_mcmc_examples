OUTDIR <- "output"

SAMPLERDIRS <- c(
  file.path(OUTDIR, "mh", "single_chain"),
  file.path(OUTDIR, "mala", "single_chain")
)

nsamplerdirs <- length(SAMPLERDIRS)

nmcmc <- 100000

i <- 8

maxlag <- 40

cors <- matrix(data=NA, nrow=maxlag+1, ncol=nsamplerdirs)

for (j in 1:nsamplerdirs) {
  chain <- read.table(file.path(SAMPLERDIRS[j], 'chain.csv'), header=FALSE, sep=',')
  cors[, j] <- acf(chain[, i], lag.max=maxlag, demean=TRUE, plot=FALSE)$acf
}

sqrtnmcmc <- sqrt(nmcmc)

cols <- c("red", "blue")

# pdf(file=file.path(OUTDIR, "acf_plot.pdf"), width=10, height=6)

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
  cex.lab=1.7,
  yaxt="n"
)

axis(
  2,
  at=seq(-0.2, 1, by=0.2),
  labels=seq(-0.2, 1, by=0.2),
  cex.axis=1.8,
  las=1
)

lines(
  0:maxlag,
  cors[, 2],
  type="o",
  col=cols[2],
  lwd=2,
  pch=20
)

abline(h=1.96/sqrtnmcmc, lty=2)
abline(h=-1.96/sqrtnmcmc, lty=2)

par(fig=c(0, 1, 0.89, 1), mar=c(0, 0, 0, 0), new=TRUE)

plot.new()

legend(
  "center",
  c("MH", "MALA"),
  lty=c(1, 1),
  lwd=c(5, 5),
  col=cols,
  cex=1.5,
  bty="n",
  text.width=0.125,
  ncol=2
)

par(oldpar)

# dev.off()
