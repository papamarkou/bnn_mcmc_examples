# Load Stat2Data library, which contains the hawks dataframe

library(Stat2Data)

# Load dataframe

data(Hawks)

# penguins.csv downloaded from
# https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv

# Dimensions of dataframe

print(c("Dimensions of dataframe:"))
print(dim(Hawks))

# Frequencies of species

print("Frequencies of species:")
print(table(Hawks$Species))

# Dimensions of dataframe without missing values

print("Dimensions of dataframe without missing values:")
print(dim(na.omit(Hawks)))

# Frequencies of species without missing values

print("Frequencies of species without missing values:")
print(table(na.omit(Hawks)$Species))

# Missing values per column

print("Missing values per column:")
print(sapply(Hawks, function(x) sum(is.na(x))))

# Keep only data subset of columns and omit associated missing values

reduced_dframe <- na.omit(Hawks[c("Species", "Age", "Wing", "Weight", "Culmen", "Hallux", "Tail")])

# Dimensions of reduced dataframe

print(c("Dimensions of reduced dataframe:"))
print(dim(reduced_dframe))

# Frequencies of species in reduced dataframe

print("Frequencies of species in reduced dataframe:")
print(table(na.omit(reduced_dframe)$Species))

# Convert factor variables to numeric variables

reduced_dframe$Species <- as.numeric(reduced_dframe$Species) - 1 # (CH, RT, SS) = (1, 2, 3)
reduced_dframe$Age <- as.numeric(reduced_dframe$Age) - 1 # (A, I) = (1, 2)

# Write reduced dataframe to CSV

write.table(reduced_dframe[, -1], "x.csv", quote=FALSE, sep=",", row.names=FALSE)
write.table(reduced_dframe[c("Species")], "y.csv", quote=FALSE, sep=",", row.names=FALSE)
