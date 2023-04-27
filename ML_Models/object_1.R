# Load required packages
library(readxl)
library(NbClust)
library(ggplot2)
library(cluster)
library(factoextra)
library(FactoMineR)

# Set the file path and sheet name
file_path <- "Data Sets/vehicles.xlsx"
sheet_name <- "vehicle"

# Load the data from the sheet into a data frame
vehicles <- read_excel(file_path, sheet = sheet_name)

# Scale the data
scaled_vehicles <- scale(vehicles[, 1:18])

# Detect and remove outliers using the Z-score method
z_scores <- apply(scaled_vehicles, 1, function(x) sum(abs(x) > 3))
outliers <- which(z_scores > 0)
scaled_vehicles <- scaled_vehicles[-outliers, ]

# Determine the optimal number of clusters using NbClust
nbclust_index <- NbClust(scaled_vehicles, min.nc = 2, max.nc = 10, method = "kmeans", index = "all")


# Plot the elbow curve
set.seed(123)
wss <- c()
for (i in 1:10) {
  km <- kmeans(scaled_vehicles, centers = i, nstart = 10)
  wss[i] <- sum(km$withinss)
}
plot(wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters", ylab = "WSS",
     main = "Elbow Method")

# Find the optimal number of clusters using elbow method
elbow_index <- which(diff(wss) < mean(diff(wss)))[1] + 2
cat("Optimal number of clusters using elbow method: ", elbow_index, "\n")

# Compute and plot the gap statistic with number of clusters
set.seed(123)
gap_stat <- clusGap(scaled_vehicles, FUN = kmeans,
                    K.max = 2, B = 50, verbose = FALSE,
                    nstart = 10)
plot(gap_stat, main = "Gap Statistic", xlab = "Number of clusters")

# Find the optimal number of clusters using gap statistic
gap_index <- which.max(gap_stat$Tab[, "gap"]) + 1
cat("Optimal number of clusters using gap statistic: ", gap_index, "\n")

# Calculate the average silhouette width for different values of k

k.min <- 2
k.max <- 10

# Create a list to store the silhouette values for each value of K
silhouette_vals <- vector("list", k.max - k.min + 1)

# Loop through each value of K and perform clustering using K-means algorithm
for (k in k.min:k.max) {
  km <- kmeans(scaled_vehicles, centers = k, nstart = 10)

  # Calculate the silhouette width for each data point
  silhouette_vals[[k - k.min + 1]] <- silhouette(km$cluster, dist(scaled_vehicles))
}

# Calculate the average silhouette width for each value of K
silhouette_avg <- sapply(silhouette_vals, function(x) mean(x[, 3]))

# Create a data frame with the silhouette widths for each value of K
df <- data.frame(k = k.min:k.max, silhouette = silhouette_avg)

# Plot the silhouette widths using ggplot2
ggplot(df, aes(x = k, y = silhouette)) +
  geom_point() +
  geom_line() +
  labs(x = "Number of clusters", y = "Silhouette")

# Find the index of the maximum silhouette width
best_k <- which.max(silhouette_avg) + k.min - 1

# Print the best number of clusters based on the silhouette method
cat("Best number of clusters based on the silhouette method: ", best_k, "\n")


# Set the optimal number of clusters using the best method (elbow) from the above
k <- elbow_index

# Perform k-means clustering
set.seed(123)
kmeans_model <- kmeans(scaled_vehicles, centers = k, nstart = 25)

# Show the k-means output
print(kmeans_model)

# Calculate the between-cluster sum of squares (BSS) and total sum of squares (TSS)
BSS <- sum(kmeans_model$size * colSums((kmeans_model$centers - colMeans(scaled_vehicles))^2))
TSS <- sum((scaled_vehicles - colMeans(scaled_vehicles))^2)

# Calculate the ratio of BSS over TSS
ratio_BSS_TSS <- BSS/TSS
cat("Ratio of between-cluster sum of squares over total sum of squares: ", ratio_BSS_TSS, "\n")

# Calculate the within-cluster sum of squares (WSS)
WSS <- sum(kmeans_model$withinss)
cat("Within-cluster sum of squares: ", WSS, "\n")

# Plot the cluster results
fviz_cluster(kmeans_model, data = scaled_vehicles)

# Compute the silhouette widths
silhou_width <- silhouette(kmeans_model$cluster, dist(scaled_vehicles))

# Plot the silhouette plot
plot(silhou_width, main = "Silhouette Plot for K-Means Clustering")

# Plot the silhouette plot in a new page
pdf("silhouette_plot.pdf")
plot(silhou_width, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters", ylab = "Silhouette Width",
     main = "Silhouette Method")
dev.off()

# Compute the silhouette widths
silhou_width <- silhouette(kmeans_model$cluster, dist(scaled_vehicles))

# Plot the silhouette plot
plot(silhou_width, main = "Silhouette Plot for K-Means Clustering")
avg_sil_width <- summary(silhou_width)$avg.width
cat("Average Silhouette Width: ", avg_sil_width, "\n")

#second sub task - clustering object

# Perform PCA on the scaled data
pca_vehicles <- prcomp(scaled_vehicles, center = TRUE, scale. = TRUE)

# View eigenvalues and eigenvectors
print(pca_vehicles)

# View cumulative scores per principal component
plot(pca_vehicles)

# Create a new dataset with principal components as attributes
transformed_vehicles <- as.data.frame(predict(pca_vehicles))

# Choose principal components that provide at least 92% cumulative score
cumulative_scores <- cumsum(pca_vehicles$sdev^2 / sum(pca_vehicles$sdev^2))
num_components <- sum(cumulative_scores <= 0.92)
chosen_components <- paste0("PC", 1:num_components)
transformed_vehicles <- transformed_vehicles[, chosen_components]

# Determine the optimal number of clusters using NbClust
nbclust_index <- NbClust(transformed_vehicles, min.nc = 2, max.nc = 10, method = "kmeans", index = "all")

