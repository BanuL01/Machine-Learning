# Load required packages
library(readxl)
library(NbClust)
library(ggplot2)
library(cluster)
library(factoextra)
library(FactoMineR)
library(fpc)

# Set the file path and sheet name
file_path <- "Data Sets/vehicles.xlsx"
sheet_name <- "vehicle"

# Load the data from the sheet into a data frame
vehicles_data <- read_excel(file_path, sheet = sheet_name)

# Scale the data
scaled_v_data <- scale(vehicles_data[, 1:18])

# Detect and remove outliers using the Z-score method
z_score <- apply(scaled_v_data, 1, function(x) sum(abs(x) > 3))
outliers <- which(z_score > 0)
scaled_v_data <- scaled_v_data[-outliers, ]

# Determine the optimal number of clusters using NbClust
nbclust_num <- NbClust(scaled_v_data, min.nc = 2, max.nc = 10, method = "kmeans", index = "all")

# Create a data frame with the clustering indices and the number of clusters
df <- data.frame(Clusters = 2:10, nbclust_num$All.index)

# Melt the data frame to long format
df_long <- reshape2::melt(df, id.vars = "Clusters", variable.name = "Index", value.name = "Value")

# Plot the bar plot using ggplot2
ggplot(df_long, aes(x = Clusters, y = Value, fill = Index)) +
  geom_bar(stat = "identity", position = "dodge") +
  xlab("Number of clusters") +
  ylab("Clustering index") +
  ggtitle("NbClust plot") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_vline(xintercept = nbclust_num$Best.nc[1], linetype = "dashed", color = "blue")


# Plot the elbow curve
set.seed(123)
wss <- c()
for (i in 1:10) {
  km <- kmeans(scaled_v_data, centers = i, nstart = 10)
  wss[i] <- sum(km$withinss)
}
plot(wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters", ylab = "WSS",
     main = "Elbow Method")

# Find the optimal number of clusters using elbow method
elbow_num <- which(diff(wss) < mean(diff(wss)))[1] + 2
cat("Optimal number of clusters using elbow method: ", elbow_num, "\n")

# Compute and plot the gap statistic with number of clusters
set.seed(123)
gap_st <- clusGap(scaled_v_data, FUN = kmeans,
                    K.max = 2, B = 50, verbose = FALSE,
                    nstart = 10)
plot(gap_st, main = "Gap Statistic", xlab = "Number of clusters")

# Find the optimal number of clusters using gap statistic
gapst_num <- which.max(gap_st$Tab[, "gap"]) + 1
cat("Optimal number of clusters using gap statistic: ", gapst_num, "\n")

# Calculate the average silhouette width for different values of k

k.min <- 2
k.max <- 10

# Create a list to store the silhouette values for each value of K
silhouette_vals <- vector("list", k.max - k.min + 1)

# Loop through each value of K and perform clustering using K-means algorithm
for (k in k.min:k.max) {
  km <- kmeans(scaled_v_data, centers = k, nstart = 10)

  # Calculate the silhouette width for each data point
  silhouette_vals[[k - k.min + 1]] <- silhouette(km$cluster, dist(scaled_v_data))
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
k <- elbow_num

# Perform k-means clustering
set.seed(123)
kmeans_model <- kmeans(scaled_v_data, centers = k, nstart = 25)

# Show the k-means output
print(kmeans_model)

# Calculate the between-cluster sum of squares (BSS) and total sum of squares (TSS)
BSS <- sum(kmeans_model$size * colSums((kmeans_model$centers - colMeans(scaled_v_data))^2))
TSS <- sum((scaled_v_data - colMeans(scaled_v_data))^2)

# Calculate the ratio of BSS over TSS
ratio_BSS_TSS <- BSS/TSS
cat("Ratio of between-cluster sum of squares over total sum of squares: ", ratio_BSS_TSS, "\n")

# Calculate the within-cluster sum of squares (WSS)
WSS <- sum(kmeans_model$withinss)
cat("Within-cluster sum of squares: ", WSS, "\n")

# Plot the cluster results
fviz_cluster(kmeans_model, data = scaled_v_data)

# Compute the silhouette widths
silhou_width <- silhouette(kmeans_model$cluster, dist(scaled_v_data))

# Plot the silhouette plot
plot(silhou_width, main = "Silhouette Plot for K-Means Clustering")

# Plot the silhouette plot in a new page
pdf("silhouette_plot.pdf")
plot(silhou_width, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters", ylab = "Silhouette Width",
     main = "Silhouette Method")
dev.off()

# Compute the silhouette widths
silhou_width <- silhouette(kmeans_model$cluster, dist(scaled_v_data))

# Plot the silhouette plot
plot(silhou_width, main = "Silhouette Plot for K-Means Clustering")
avg_sil_width <- summary(silhou_width)$avg.width
cat("Average Silhouette Width: ", avg_sil_width, "\n")

#second sub task - clustering object

# # Perform PCA on the scaled data
# pca_vehicles <- prcomp(scaled_v_data, center = TRUE, scale. = TRUE)
# 
# # View eigenvalues and eigenvectors
# print(pca_vehicles)
# 
# # View cumulative scores per principal component
# plot(pca_vehicles)
# 
# # Create a new dataset with principal components as attributes
# transformed_vehicles <- as.data.frame(predict(pca_vehicles))
# 
# # Choose principal components that provide at least 92% cumulative score
# cumulative_scores <- cumsum(pca_vehicles$sdev^2 / sum(pca_vehicles$sdev^2))
# num_components <- sum(cumulative_scores <= 0.92)
# chosen_components <- paste0("PC", 1:num_components)
# transformed_vehicles <- transformed_vehicles[, chosen_components]
# 
# # Determine the optimal number of clusters using NbClust
# nbclust_num_trans <- NbClust(transformed_vehicles, min.nc = 2, max.nc = 10, method = "kmeans", index = "all")
# 
# 
# # Plot the elbow curve 
# 
# set.seed(123)
# wss <- c()
# for (i in 1:10) {
#   km <- kmeans(transformed_vehicles, centers = i, nstart = 10)
#   wss[i] <- sum(km$withinss)
# }
# plot(wss, type = "b", pch = 19, frame = FALSE,
#      xlab = "Number of clusters", ylab = "WSS",
#      main = "Elbow Method")
# 
# # Find the optimal number of clusters using elbow method
# elbow_num_trans <- which(diff(wss) < mean(diff(wss)))[1] + 2
# cat("Optimal number of clusters using elbow method: ", elbow_num_trans, "\n")
# 
# # Compute and plot the gap statistic with number of clusters
# set.seed(123)
# gap_st_trans <- clusGap(transformed_vehicles, FUN = kmeans,
#                   K.max = 2, B = 50, verbose = FALSE,
#                   nstart = 10)
# plot(gap_st_trans, main = "Gap Statistic", xlab = "Number of clusters")
# 
# # Find the optimal number of clusters using gap statistic
# gapst_num_trans <- which.max(gap_st_trans$Tab[, "gap"]) + 1
# cat("Optimal number of clusters using gap statistic: ", gapst_num_trans, "\n")
# 
# 
# # Create a list to store the silhouette values for each value of K
# silhouette_vals_trans<- vector("list", 10 - 2 + 1)
# 
# # Loop through each value of K and perform clustering using K-means algorithm
# for (k in 2:10) {
#   km <- kmeans(transformed_vehicles, centers = k, nstart = 10)
#   
#   # Calculate the silhouette width for each data point
#   silhouette_vals_trans[[k - 2 + 1]] <- silhouette(km$cluster, dist(transformed_vehicles))
# }
# 
# # Calculate the average silhouette width for each value of K
# silhouette_avg_trans <- sapply(silhouette_vals_trans, function(x) mean(x[, 3]))
# 
# # Create a data frame with the silhouette widths for each value of K
# df_trans <- data.frame(k = 2:10, silhouette = silhouette_avg_trans)
# 
# # Plot the silhouette widths using ggplot2
# ggplot(df_trans, aes(x = k, y = silhouette)) +
#   geom_point() +
#   geom_line() +
#   labs(x = "Number of clusters", y = "Silhouette")
# 
# # Find the index of the maximum silhouette width
# best_k_trans <- which.max(silhouette_avg_trans) + 2 - 1
# 
# # Print the best number of clusters based on the silhouette method
# cat("Best number of clusters based on the silhouette method: ", best_k_trans, "\n")
# 
# # Set the optimal number of clusters using the best method (elbow) from the above
# k <- elbow_num_trans
# 
# # Perform k-means clustering
# set.seed(123)
# kmeans_model_trans <- kmeans(transformed_vehicles, centers = k, nstart = 25)
# 
# # Show the k-means output
# print(kmeans_model_trans)
# 
# # Calculate the between-cluster sum of squares (BSS) and total sum of squares (TSS)
# BSS_trans <- sum(kmeans_model_trans$size * colSums((kmeans_model_trans$centers - colMeans(transformed_vehicles))^2))
# TSS_trans <- sum((transformed_vehicles - colMeans(transformed_vehicles))^2)
# 
# # Calculate the ratio of BSS over TSS
# ratio_BSS_TSS_trans <- BSS_trans/TSS_trans
# cat("Ratio of between-cluster sum of squares over total sum of squares: ", ratio_BSS_TSS_trans, "\n")
# 
# # Calculate the within-cluster sum of squares (WSS)
# WSS_trans  <- sum(kmeans_model_trans$withinss)
# cat("Within-cluster sum of squares: ", WSS_trans, "\n")
# 
# # Plot the cluster results
# fviz_cluster(kmeans_model_trans, data = transformed_vehicles)
# 
# # Compute the silhouette widths
# silhou_width_trans <- silhouette(kmeans_model_trans$cluster, dist(transformed_vehicles))
# 
# # Plot the silhouette plot
# plot(silhou_width_trans, main = "Silhouette Plot for K-Means Clustering")
# 
# # Plot the silhouette plot in a new page
# pdf("silhouette_plot.pdf")
# plot(silhou_width_trans, type = "b", pch = 19, frame = FALSE,
#      xlab = "Number of clusters", ylab = "Silhouette Width",
#      main = "Silhouette Method")
# dev.off()
# 
# 
# # Plot the silhouette plot
# plot(silhou_width_trans, main = "Silhouette Plot for K-Means Clustering")
# avg_sil_width_trans <- summary(silhou_width_trans)$avg.width
# cat("Average Silhouette Width: ", avg_sil_width_trans, "\n")
# 
# # Compute dissimilarity matrix
# dissim_mat <- dist(transformed_vehicles)
# 
# # Compute Calinski-Harabasz index
# cal_h_index <- cluster.stats(dissim_mat, kmeans_model_trans$cluster)$ch
# # Plot the CH index
# cal_h_data <- data.frame(K = 1:length(cal_h_index), CH_Index = cal_h_index)
# 
# ggplot(cal_h_data, aes(x = K, y = CH_Index)) +
#   geom_point(color = "blue") +
#   labs(x = "Number of Clusters (K)", y = "Calinski-Harabasz Index") +
#   ggtitle("Calinski-Harabasz Index for K-means Clustering Results")
# 
# # Print the CH index
# cat("Calinski-Harabasz Index: ", cal_h_index, "\n")