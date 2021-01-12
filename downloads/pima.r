# Load mlbench library, which contains the Pima dataframe

library(mlbench)

# Load Pima dataframe

data(PimaIndiansDiabetes2)

# Set the seed for splitting the data 

# print("Setting the seed")
# set.seed(100)

# Dimensions of Pima dataframe

print("Dimensions of dataframe:")
print(dim(PimaIndiansDiabetes2))
dim(na.omit(PimaIndiansDiabetes2))

# Check for missing values

print("Checking for missing values:")
print(table(is.na(PimaIndiansDiabetes2))) # dataframe-wise
# sapply(PimaIndiansDiabetes2, function(x) sum(is.na(x))) # column-wise
# rowSums(is.na(PimaIndiansDiabetes2)) # row-wise

# 

model <- glm(
  diabetes ~ pregnant + glucose + pressure + triceps + insulin + mass + pedigree + age,
  data=PimaIndiansDiabetes2,
  family="binomial"
)

print(summary(model))

filtered_dframe <- PimaIndiansDiabetes2[c("glucose", "mass", "pedigree", "age", "diabetes")]

summary(glm(
  diabetes ~  glucose + mass + pedigree + age,
  data=PimaIndiansDiabetes2,
  family="binomial"
))

# Frequency of diabetes cases

print("Frequency of diabetes cases:")
print(table(PimaIndiansDiabetes2$diabetes))

#

dframe <- PimaIndiansDiabetes2
dframe$diabetes <- ifelse(PimaIndiansDiabetes2$diabetes == "pos", 1, 0)
