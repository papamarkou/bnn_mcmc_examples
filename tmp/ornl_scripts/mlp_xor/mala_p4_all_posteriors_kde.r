library(bde)

OUTDIR <- file.path("output", "mala", "multiple_chains")
OUTDIR01 <- file.path("output", "mala", "multiple_chains", "normal_prior", "mean10")
OUTDIR02 <- file.path("output", "mala", "multiple_chains", "normal_prior", "var003")
OUTDIR03 <- file.path("output", "mala", "multiple_chains", "normal_prior", "var010")
OUTDIR04 <- file.path("output", "mala", "multiple_chains", "normal_prior", "var100")
OUTDIR05 <- file.path("output", "mala", "multiple_chains", "uniform_prior", "a020", "drift2p49")

store <- TRUE

nmcmc <- 100000

chain_id <- c(1, 1, 1, 1, 3) # Chain
j <- 4 # Coordinate

chain01 <- read.table(
  file.path(OUTDIR01, paste('chain', str_pad(chain_id[1], 2, pad='0'), '.csv', sep='')), header=FALSE, sep=","
)
chain02 <- read.table(
  file.path(OUTDIR02, paste('chain', str_pad(chain_id[2], 2, pad='0'), '.csv', sep='')), header=FALSE, sep=","
)
chain03 <- read.table(
  file.path(OUTDIR03, paste('chain', str_pad(chain_id[3], 2, pad='0'), '.csv', sep='')), header=FALSE, sep=","
)
chain04 <- read.table(
  file.path(OUTDIR04, paste('chain', str_pad(chain_id[4], 2, pad='0'), '.csv', sep='')), header=FALSE, sep=","
)
chain05 <- read.table(
  file.path(OUTDIR05, paste('chain', str_pad(chain_id[5], 2, pad='0'), '.csv', sep='')), header=FALSE, sep=","
)

chain_kde01 <- density(chain01[, j])
chain_kde02 <- density(chain02[, j])
chain_kde03 <- density(chain03[, j])
chain_kde04 <- density(chain04[, j])
chain_kde05 <- chen99Kernel(chain05[, j], lower.limit=-20, upper.limit=20, b=0.01)

cols <- c("green", "blue", "purple", "orange", "red")

if (store) {
  pdf(file=file.path(OUTDIR, paste("mala_p", j,"_all_posteriors_kde.pdf", sep='')), width=10, height=6)
}

oldpar <- par(no.readonly=TRUE)

par(fig=c(0, 1, 0, 1), mar=c(2.25, 4, 3.5, 1)+0.1)

plot(
  chain_kde01$x,
  chain_kde01$y,
  type="l",
  xlim=c(-20, 20),
  ylim=c(0, 0.24),
  col=cols[1],
  xlab="Iteration",
  ylab="",
  # main=bquote(paste(theta[.(j)], ": N(0,3I) prior", sep=" ")),
  main="",
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
  at=seq(0, 0.24, by=0.06),
  labels=seq(0, 0.24, by=0.06),
  cex.axis=1.8,
  las=1
)

lines(
  chain_kde02$x,
  chain_kde02$y,
  type="l",
  col=cols[2],
  lwd=2
)

lines(
  chain_kde03$x,
  chain_kde03$y,
  type="l",
  col=cols[3],
  lwd=2
)

lines(
  chain_kde04$x,
  chain_kde04$y,
  type="l",
  col=cols[4],
  lwd=2
)

lines(
  chain_kde05,
  type="l",
  col=cols[5],
  lwd=2
)

par(fig=c(0, 1, 0.89, 1), mar=c(0, 0, 0, 0), new=TRUE)

plot.new()

sampler_labels <- c(
  "MALA, N(10, 3I)",
  "MALA, N(0, 3I)",
  "MALA, N(0, 10I)",
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
