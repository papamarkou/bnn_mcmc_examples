# Load mlbench library, which contains the Pima dataframe
library(mlbench)

# Check for missing values
# table(is.na(PimaIndiansDiabetes)) # dataframe-wise
# sapply(PimaIndiansDiabetes, function(x) sum(is.na(x))) # column-wise
# rowSums(is.na(PimaIndiansDiabetes)) # row-wise

# Frequency of diabetes cases
# table(PimaIndiansDiabetes$diabetes)

out_frame <- PimaIndiansDiabetes
out_frame$diabetes <- ifelse(PimaIndiansDiabetes$diabetes == "pos", 1, 0)
