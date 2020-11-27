# Dimensions 4, 7, 8
# For dimension 4, chains 1, 2, 3, 4, 6, 7, 8, 9, 10
# plot(as.matrix(mcchains[[10]])[50000:100000, 4], type="l")

# For dimension 7, chains 5, 6, 7, 9
plot(as.matrix(mcchains[[9]])[50000:100000, 7], type="l")

apply(as.matrix(mcchains[[9]]), 2, mean)
# V1        V2        V3        V4        V5        V6        V7        V8        V9 
# 10.012597  9.993248 10.028609 10.014103  9.981904  9.993595  4.013294  4.007706  4.005025
