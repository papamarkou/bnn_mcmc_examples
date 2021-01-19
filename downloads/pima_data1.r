# Load mlbench library, which contains the Pima dataframe

library(mlbench)

# Load dataframe

data(PimaIndiansDiabetes2)

# Dimensions of dataframe

print(c("Dimensions of dataframe:"))
print(dim(PimaIndiansDiabetes2))

# Frequency of diabetes cases

print("Frequency of diabetes cases:")
print(table(PimaIndiansDiabetes2$diabetes))

# Dimensions of dataframe without missing values

print("Dimensions of dataframe without missing values:")
print(dim(na.omit(PimaIndiansDiabetes2)))

# Frequency of diabetes cases without missing values

print("Frequency of diabetes cases without missing values:")
print(table(na.omit(PimaIndiansDiabetes2)$diabetes))

# Missing values per column

print("Missing values per column:")
print(sapply(PimaIndiansDiabetes2, function(x) sum(is.na(x))))

# Entries with missing values in the dataframe

# print("Entries with missing values in the dataframe:")
# print(table(is.na(PimaIndiansDiabetes2)))

# Missing values per row

# print("Missing values per row:")
# print(rowSums(is.na(PimaIndiansDiabetes2)))

# Keep only data without missing values

reduced_dframe <- na.omit(PimaIndiansDiabetes2)

# Fit logistic regression to the data

model <- glm(
  diabetes ~ pregnant + glucose + pressure + triceps + insulin + mass + pedigree + age,
  data=reduced_dframe,
  family="binomial"
)

# Print summary of logistic regression

print("Summary of logistic regression:")
print(summary(model))

# Convert factor output variable to numeric

reduced_dframe$diabetes <- ifelse(reduced_dframe$diabetes == "pos", 1, 0)

# Dimensions of reduced dataframe

print(c("Dimensions of reduced dataframe:"))
print(dim(reduced_dframe))

# Frequency of diabetes cases in reduced dataframe

print("Frequency of diabetes cases in reduced dataframe:")
print(table(reduced_dframe$diabetes))

# Write reduced dataframe to CSV

write.table(reduced_dframe[, -9], "x.csv", quote=FALSE, sep=",", row.names=FALSE)
write.table(reduced_dframe[c("diabetes")], "y.csv", quote=FALSE, sep=",", row.names=FALSE)
