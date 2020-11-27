OUTDIR <- "output"

# library("freqparcoord")

thetas <- read.table(file.path(OUTDIR, "optim", "multiple_runs", "normal_prior", "var010", "iters010000", "thetas.csv"), header=FALSE, sep=',')

num_samples <- 10000
num_pars <- ncol(thetas)

for (i in seq(1, num_pars)) {
  hist(
    thetas[, i],
    breaks=50,
    freq=FALSE,
    main=paste("Parameter ", i, sep=''),
    xlab="Value range",
    ylab="Relative frequency"
  )
}

theta_maxs <- apply(thetas, 2, max)
theta_mins <- apply(thetas, 2, min)

plot(seq(1, num_pars), as.matrix(thetas[1, ]), ylim=c(min(theta_mins), max(theta_maxs)), type="l")
for (i in seq(2, num_samples)) {
  lines(seq(1, num_pars), as.matrix(thetas[i, ]))
}

# https://cran.r-project.org/web/packages/freqparcoord/freqparcoord.pdf
# gg <- freqparcoord(thetas, num_samples)
# gg <- freqparcoord(thetas, num_samples)
# gg + theme_classic() + theme(plot.margin = margin(0.5, .8, 2, .8, "cm"))
