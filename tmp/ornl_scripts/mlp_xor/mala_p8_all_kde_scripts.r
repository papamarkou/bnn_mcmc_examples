scripts <- c(
  "mala_p8_all_posteriors_kde.r",
  "mala_p8_normal_prior_mean10_kde.r",
  "mala_p8_normal_prior_var003_kde.r",
  "mala_p8_normal_prior_var010_kde.r",
  "mala_p8_normal_prior_var100_kde.r",
  "mala_p8_uniform_prior_a020_kde.r"
)

print(getwd())

for (s in scripts) {
  object_names <- ls(all.names = TRUE)
  object_names <- setdiff(object_names, c("s", "scripts"))
  rm(object_names)
  print(paste("Executing", s))
  source(s)
}

rm(list = ls(all.names = TRUE))
