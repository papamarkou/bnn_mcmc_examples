# penguins.csv downloaded from
# https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv

# Load dataframe

dframe <- read.csv(file="penguins.csv", header=TRUE)

# Dimensions of dataframe

print(c("Dimensions of dataframe:"))
print(dim(dframe))

# Frequencies of species

print("Frequencies of species:")
print(table(dframe$species))

# Dimensions of dataframe without missing values

print("Dimensions of dataframe without missing values:")
print(dim(na.omit(dframe)))

# Frequency of diabetes cases without missing values

print("Frequencies of species without missing values:")
print(table(na.omit(dframe)$species))

# Missing values per column

print("Missing values per column:")
print(sapply(dframe, function(x) sum(is.na(x))))

# Keep only data without missing values

reduced_dframe <- na.omit(dframe)

# Convert factor variables to numeric variables

reduced_dframe$species <- as.numeric(reduced_dframe$species) # (Adelie, Chinstrap, Gentoo) = (1, 2, 3)
reduced_dframe$island <- as.numeric(reduced_dframe$island) # (Biscoe, Dream, Torgersen) = (1, 2, 3)
reduced_dframe$sex <- as.numeric(reduced_dframe$sex) # (female, male) = (1, 2)

# Write reduced dataframe to CSV

write.table(reduced_dframe[, -1], "x.csv", quote=FALSE, sep=",", row.names=FALSE)
write.table(reduced_dframe[c("species")], "y.csv", quote=FALSE, sep=",", row.names=FALSE)
