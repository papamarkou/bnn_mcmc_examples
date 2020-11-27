recursive_mean <- function(lastmean, k, x){
  return(((k-1)*lastmean+x)/k)
}

OUTDIR <- "output"

SAMPLERDIRS <- c(
  file.path(OUTDIR, "mh", "single_chain"),
  file.path(OUTDIR, "mala", "single_chain")
)

nsamplerdirs <- length(SAMPLERDIRS)

nmcmc <- 100000

i <- 8

submeans <- matrix(data=NA, nrow=nmcmc, ncol=nsamplerdirs)
curmeans <- rep(0, nsamplerdirs)

for (j in 1:nsamplerdirs) {
  chain <- read.table(file.path(SAMPLERDIRS[j], 'chain.csv'), header=FALSE, sep=',')

  for (k in 1:nmcmc) {
    curmeans[j] <- recursive_mean(curmeans[j], k, chain[k, i])
    submeans[k, j] <- curmeans[j]
  }
}

cols <- c("red", "blue")

# pdf(file=file.path(OUTDIR, "mean_plot.pdf"), width=10, height=6)

oldpar <- par(no.readonly=TRUE)

par(fig=c(0, 1, 0, 1), mar=c(2.25, 4, 3.5, 1)+0.1)

plot(
  1:nmcmc,
  submeans[, 1],
  type="l",
  ylim=c(-0.75, 1.75),
  col=cols[1],
  lwd=2,
  xlab="",
  ylab="",
  cex.axis=1.8,
  cex.lab=1.7,
  yaxt="n"
)

axis(
  2,
  at=seq(-0.5, 1.5, by=0.5),
  labels=seq(-0.5, 1.5, by=0.5),
  cex.axis=1.8,
  las=1
)

lines(
  1:nmcmc,
  submeans[, 2],
  type="l",
  col=cols[2],
  lwd=2
)

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
