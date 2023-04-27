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

# Display the results
print(nbclust_index)

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

# Compute and plot the silhouette widths
set.seed(123)
sil_width <- c()
for (i in 2:10) {
  km <- kmeans(scaled_vehicles, centers = i, nstart = 10)
  sil_width[i] <- mean(silhouette(km$cluster, dist(scaled_vehicles)))
}
plot(sil_width, type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of clusters", ylab = "Silhouette Width", 
     main = "Silhouette Method")
cat("Optimal number of clusters using silhouette method: ", which.max(sil_width), "\n")

# Set the optimal number of clusters
k <- gap_index

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

# Display the optimal number of clusters using silhouette method
cat("Optimal number of clusters using silhouette method: ", which.max(sil_width), "\n")

# Compute the silhouette widths
silhou_width <- silhouette(kmeans_model$cluster, dist(scaled_vehicles))

# Plot the silhouette plot
plot(silhou_width, main = "Silhouette Plot for K-Means Clustering")
avg_sil_width <- summary(silhou_width)$avg.width
cat("Average Silhouette Width: ", avg_sil_width, "\n")


# Load the iris dataset
data(iris)

# Extract the numerical variables
iris_num <- iris[, 1:4]

# Create the data matrix
data_matrix <- as.matrix(iris_num)

pca_res <- prcomp(data_matrix)

# Check number of columns in pca_res$ind$coord
if (ncol(pca_res$ind$coord) < 6) {
  # If number of columns is less than 6, use all columns
  coord_cols <- 1:ncol(pca_res$ind$coord)
} else {
  # If number of columns is greater than or equal to 6, use first 6 columns
  coord_cols <- 1:6
}

# Extract PCA coordinates for first 6 components
pca_coords <- pca_res$ind$coord[, coord_cols]


# Perform PCA analysis
pca_res <- PCA(scaled_vehicles, graph = FALSE)
# Perform PCA

# Display the eigenvalues and eigenvectors
print(pca_res$eig)
print(pca_res$var$contrib)

# Plot the cumulative score per principal components
fviz_eig(pca_res, addlabels = TRUE, ylim = c(0, 100))

# Create a transformed dataset with principal components as attributes
transformed_vehicles <- as.data.frame(pca_res$ind$coord[, 1:6])

# Choose those PCs that provide at least cumulative score > 92%
pc_cumsum <- cumsum(pca_res$eig$percent)
chosen_pcs <- which(pc_cumsum > 92)

# Provide a brief discussion for your choice to choose specific number of PCs
cat("Cumulative percentage of variance explained by each PC:\n")
print(pc_cumsum)
cat("\n")
cat("Number of PCs needed to explain at least 92% of the variance: ", length(chosen_pcs), "\n")


