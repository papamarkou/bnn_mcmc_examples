library(bde) # For kernel density estimation for density of bounded support

OUTDIR00 <- file.path("output", "optim", "multiple_runs")
OUTDIR01 <- file.path("output", "optim", "multiple_runs", "normal_prior", "var010", "iters010000")
OUTDIR02 <- file.path("output", "optim", "multiple_runs", "uniform_prior", "a020", "iters010000")

store <- TRUE

thetas01 <- read.table(file.path(OUTDIR01, "thetas.csv"), header=FALSE, sep=',')
thetas02 <- read.table(file.path(OUTDIR02, "thetas.csv"), header=FALSE, sep=',')

num_samples <- 10000
num_pars <- ncol(thetas01)

j <- 4 # Coordinate

thetas_kde01 <- density(thetas01[, j])
# thetas_kde02 <- density(thetas02[, j])
thetas_kde02 <- chen99Kernel(
  thetas02[-20 < thetas02[, j] & thetas02[, j] < 20, j], lower.limit=-20, upper.limit=20, b=0.02
)

cols <- c("red", "blue")

if (store) {
  pdf(file=file.path(OUTDIR00, "optim_multiple_runs_a.pdf"), width=10, height=6)
}

oldpar <- par(no.readonly=TRUE)

par(fig=c(0, 1, 0, 1), mar=c(2.25, 4, 3.5, 1)+0.1)

plot(
  thetas_kde01$x,
  thetas_kde01$y,
  type="l",
  xlim=c(-20, 20),
  ylim=c(0, 0.16),
  col=cols[1],
  xlab="Iteration",
  ylab="",
  main=bquote(paste(theta[.(j)], ": kernel density estimates of optimization solutions", sep=" ")),
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
  at=seq(0, 0.16, by=0.04),
  labels=seq(0, 0.16, by=0.04),
  cex.axis=1.8,
  las=1
)

lines(
  thetas_kde02,
  type="l",
  col=cols[2],
  lwd=2
)

# Two modes aligned between two optimizations with different priors

legend(
  "topright",
  c("N(0, 10I)", "U(-20, 20)"),
  lty=c(1, 1),
  lwd=c(5, 5),
  seg.len=1.4,
  col=cols,
  cex=1.8,
  bty="n"
)

par(oldpar)

if (store) {
  dev.off()
}
