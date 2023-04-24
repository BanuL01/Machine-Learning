
library(readxl)
library(cluster)
library(NbClust)



#Part 1

#load vehicle data set

vehicles <- read_xlsx("Data Sets/vehicles.xlsx")
print(vehicles)

# Select the first 18 attributes
vehicles <- vehicles[,1:18]

# Scale the data using z-score normalization
vehicles_scaled <- scale(vehicles)

# Detect outliers using the z-score method
z_scores <- apply(vehicles_scaled, 2, function(x) abs(scale(x, center = TRUE, scale = FALSE)))
outliers <- apply(z_scores, 1, max) > 3

# Remove outliers
vehicles_scaled_no_outliers <- vehicles_scaled[!outliers,]

# Determine the optimal number of clusters using NbClust
set.seed(123)
nc <- NbClust(vehicles_scaled_no_outliers, min.nc = 2, max.nc = 10, method = "kmeans", index = "all")


# Determine the optimal number of clusters
best_nc <- nc$Best.nc
print(best_nc)

