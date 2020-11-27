OUTDIR00 <- file.path("output", "optim", "multiple_runs")
OUTDIR01 <- file.path("output", "optim", "multiple_runs", "normal_prior", "var010", "sigmoid_activation", "iters001000")

store <- TRUE

thetas <- read.table(file.path(OUTDIR01, "thetas.csv"), header=FALSE, sep=',')

num_samples <- 100
num_pars <- ncol(thetas)

theta_maxs <- apply(thetas[1:num_samples, ], 2, max)
theta_mins <- apply(thetas[1:num_samples, ], 2, min)

cols <- c("cornsilk4")

if (store) {
  pdf(file=file.path(OUTDIR00, "mlp_iris_optim_multiple_runs_b_i.pdf"), width=20, height=6)
}

oldpar <- par(no.readonly=TRUE)

par(fig=c(0, 1, 0, 1), mar=c(5.25, 5, 3.5, 1)+0.1)

plot(
  seq(1, num_pars),
  as.matrix(thetas[1, ]), 
  ylim=c(min(theta_mins), max(theta_maxs)),
  type="l",
  xlab="Parameter coordinate",
  ylab="Parameter value",
  main="Optimization solutions with sigmoid activation",
  col=cols[1],
  cex.axis=1.8,
  cex.main=2,
  cex.lab=2,
  lwd=1,
  xaxt="n",
  yaxt="n"
)

for (i in seq(2, num_samples)) {
  lines(seq(1, num_pars), as.matrix(thetas[i, ]), lwd=1, col=cols[1])
}

axis(
  1,
  at=c(1, seq(3, 27, by=3)),
  labels=c(1, seq(3, 27, by=3)),
  cex.axis=1.8,
  las=1
)

axis(
  2,
  at=seq(-20, 20, by=5),
  labels=seq(-20, 20, by=5),
  cex.axis=1.8,
  las=1
)

par(oldpar)

if (store) {
  dev.off()
}
