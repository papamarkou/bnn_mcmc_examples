OUTDIR <- file.path("output", "mala", "single_chain")

nmcmc <- 100000

i <- 8

chain <- read.table(file.path(OUTDIR, 'chain.csv'), header=FALSE, sep=',')

chainmean <- mean(chain[, pi])

# pdf(file=file.path(OUTDIR, "mh_traceplot.pdf"), width=10, height=6)

plot(
  1:nmcmc,
  chain[, i],
  type="l",
  # ylim=c(-1.2, 3),
  col="steelblue2",
  xlab="",
  ylab="",
  cex.axis=1.8,
  cex.lab=1.7 # ,
  # yaxt="n"
)

# axis(
#   2,
#   at=seq(-1, 3, by=1),
#   labels=seq(-1, 3, by=1),
#   cex.axis=1.8,
#   las=1
# )

# lines(
#   1:nmcmc,
#   rep(chainmean, nmcmc),
#   type="l",
#   col="black",
#   lwd=2
# )

# dev.off()
