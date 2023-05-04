#Import library
library(readxl)
library(neuralnet)
library(ggplot2)
library(MLmetrics)
library(keras)
library(dplyr)

#Part 1
# Define the root-mean-square error (RMSE) function
rmse <- function(error) {
  return(sqrt(mean(error^2)))
}

# Define the mean absolute error (MAE) function
mae <- function(error) {
  return(mean(abs(error)))
}

# Define the mean absolute percentage error (MAPE) function
mape <- function(actual, predicted) {
  return(mean(abs((actual - predicted)/actual)) * 100)
}

# Define the symmetric mean absolute percentage error (sMAPE) function
smape <- function(actual, predicted) {
  return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100)
}


# Load the UOW consumption dataset
consump_dataset<- read_xlsx("Data Sets/uow_consumption.xlsx")

# normalize the dataset to train
normalize_dataset<- as.data.frame(scale(consump_dataset[, -1]))

# add target variable back to the normalized dataset
normalize_dataset$Y <- consump_dataset$Y

# view the normalized dataset
head(normalize_dataset)

# Get summary for the column
summary(consump_dataset)


# Extract the hourly electricity consumption data for 20:00 for 2018 and 2019
#   "0.83333333333333337"= 20.00  (20/24 = 0.83333333333333337)
hourly_consump_20 <- consump_dataset[c("date", "0.83333333333333337")]

# plot the hourly consumption data for 20:00 for 2018 and 2019
ggplot(consump_dataset, aes(x=date, y=`0.83333333333333337`)) +
  geom_line() +
  labs(title = "Hourly Consumption for 20:00",
       x = "Date",
       y = "Consumption")


# Extract the first 380 samples as training data, and the remaining samples as testing data
training_data <- unlist(hourly_consump_20[1:380, "0.83333333333333337"])
testing_data <- unlist(hourly_consump_20[381:nrow(hourly_consump_20), "0.83333333333333337"])

# plot the first 380 samples as training data
ggplot(data.frame(training_data), aes(x=1:length(training_data), y=training_data)) +
  geom_line() +
  labs(title = "Training Data",
       x = "Sample Number",
       y = "Consumption")


# Define the number of time-delayed inputs
num_td_inputs <- 60


# Construct the input/output matrix for MLP training/testing
i_o_matrix <- matrix(0, nrow=length(training_data)-num_td_inputs, ncol=num_td_inputs+1)

for (i in 1:(length(training_data)-num_td_inputs)) {
  i_o_matrix[i, 1:num_td_inputs] <- training_data[i:(i+num_td_inputs-1)]
  i_o_matrix[i, num_td_inputs+1] <- training_data[i+num_td_inputs]
}

# Normalize the input/output matrix
i_o_matrix <- apply(i_o_matrix, 2, function(x) (x - mean(x)) / sd(x))

# Define the neural network structures to be evaluated
nn_structures <- list(
  c(10),
  c(20),
  c(10, 5),
  c(15, 10),
  c(15, 10, 5),
  c(25),
  c(25, 15),
  c(25, 15, 10),
  c(35),
  c(35, 25),
  c(35, 25, 15),
  c(55),
  c(55, 35),
  c(55, 35, 25)
)

results <- list()

for (i in 1:length(nn_structures)) {

  # Train the MLP using the normalized input/output matrix
  mlp <- neuralnet(V2 ~ ., data=i_o_matrix, hidden=nn_structures[[i]], linear.output=TRUE)

  # Extract the inputs for the test data
  test_data_inputs <- matrix(testing_data[1:(length(testing_data)-num_td_inputs)], ncol=num_td_inputs, byrow=TRUE)

  # Predict the output values for the test data
  mlp_output <- predict(mlp, test_data_inputs)

  # Denormalize the predicted output values
  mlp_output <- (mlp_output * sd(training_data)) + mean(training_data)

  # Calculate the MAE for the predicted and the actual output values
  mae_result <- mae(mlp_output - testing_data[(num_td_inputs+1):length(testing_data)])

  cat("the test performances for c(",nn_structures[[i]],")\n")

  # Print the MAE result
  cat("The MAE for the test data is:", round(mae_result, 2),"\n")

  # Calculate the RMSE for the predicted output values and the actual output values
  rmse_result <- rmse(mlp_output - testing_data[(num_td_inputs+1):length(testing_data)])

  # Print the RMSE result
  cat("The RMSE for the test data is:", round(rmse_result, 2),"\n")

  # Define the mean absolute percentage error (MAPE) function
  mape <- function(actual, predicted) {
    return(mean(abs((actual - predicted)/actual)) * 100)
  }

  # Calculate the MAPE for the predicted output values and the actual output values
  mape_result <- mape(testing_data[(num_td_inputs+1):length(testing_data)], mlp_output)

  # Print the MAPE result
  cat("The MAPE for the test data is:", round(mape_result, 2),"\n")

  # Define the symmetric mean absolute percentage error (sMAPE) function
  smape <- function(actual, predicted) {
    return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100)
  }

  # Calculate the sMAPE for the predicted output values and the actual output values
  smape_result <- smape(testing_data[(num_td_inputs+1):length(testing_data)], mlp_output)

  # Print the sMAPE result
  cat("The sMAPE for the test data is:", round(smape_result, 2),"\n\n")

  # Store the results for the current neural network structure
  results[[i]] <- c(nn_structures[[i]], mae_result, rmse_result, mape_result, smape_result)

}

# Create a data frame of the results
results_dframe <- data.frame(matrix(unlist(results), ncol=5, byrow=TRUE))
colnames(results_dframe) <- c("Structure", "MAE", "RMSE", "MAPE (%)", "sMAPE (%)")

# Print the comparison table of testing performances
print(results_dframe)

# Find the best one-hidden and two-hidden layer structures based on MAE and total number of weights
best_one_hid <- results_dframe[which.min(results_dframe$MAE & results_dframe$Structure),]
best_two_hid <- results_dframe[which.min(results_dframe$MAE & results_dframe$Structure),]

# Print the results
cat("Based on the comparison table, the best one-hidden layer neural network structure is",
    paste0("c(", best_one_hid$Structure, ")"),
    "with a MAE of", best_one_hid$MAE,
    "and a total number of", best_one_hid$Structure + 1, "*1+1*1=", best_one_hid$Structure + 2, "weight parameters.\n")
cat("The best two-hidden layer neural network structure is",
    paste0("c(", best_two_hid$Structure, ")"),
    "with a MAE of", best_two_hid$MAE,
    "and a total number of", sum(best_two_hid$Structure) + length(best_two_hid$Structure) + 1, "*1+1*1=", sum(best_two_hid$Structure) + length(best_two_hid$Structure) + 2, "weight parameters.\n")



#Part 2


# Define a function to build a neural network model
build_nn_model <- function(training_data, testing_data, input_vars, hidden_structure) {
  
  # Create formula for the neural network
  formula_nn <- paste("hour_20 ~", paste(input_vars, collapse = " + "))
  
  # Build the neural network model using the neural net package
  nn_model <- neuralnet(as.formula( formula_nn), training_data, hidden = hidden_structure)
  
  # Prepare the test data for prediction
  matrics_test <- as.matrix(testing_data[, input_vars, drop = FALSE])
  colnames( matrics_test) <- colnames(training_data[, input_vars, drop = FALSE])
  
  # Make predictions using the neural network model
  predictions <- predict(nn_model,  matrics_test)
  
  # Return the neural network model and its predictions
  return(list(model = nn_model, predictions = predictions))
}

# Function to calculate different evaluation metrics
calculate_metrics <- function(actual_values, predicted_values) {
  # Calculate Root Mean Squared Error
  rmse <- sqrt(mean((actual_values - predicted_values)^2))
  
  # Calculate Mean Absolute Error
  mae <- mean(abs(actual_values - predicted_values))
  
  # Calculate Mean Absolute Percentage Error
  mape <- mean(abs((actual_values - predicted_values) / actual_values)) * 100
  
  # Calculate Symmetric Mean Absolute Percentage Error
  smape <- mean(abs(actual_values - predicted_values) / (abs(actual_values) + abs(predicted_values)) * 2) * 100
  
  # Return a list containing all the evaluation metrics
  return(list(RMSE = rmse, MAE = mae, MAPE = mape, sMAPE = smape))
}

# Denormalize the predictions
denormalize <- function(x, min_value, max_value) {
  return(x * (max_value - min_value) + min_value)
}


# Rename columns to be more descriptive
colnames(consump_dataset) <- c("date", "hour_18", "hour_19", "hour_20")

# Create lagged variables for hour_20, with different time lags
consump_dataset$lag_1 <- lag(consump_dataset$hour_20, 1)
consump_dataset$lag_2 <- lag(consump_dataset$hour_20, 2)
consump_dataset$lag_3 <- lag(consump_dataset$hour_20, 3)
consump_dataset$lag_4 <- lag(consump_dataset$hour_20, 4)
consump_dataset$lag_7 <- lag(consump_dataset$hour_20, 7)

# Remove rows with missing values
consump_dataset <- na.omit(consump_dataset)

# Split data into training and testing sets based on row index
training <- consump_dataset[1:380,]
testing <- consump_dataset[381:nrow(consump_dataset),]

# Define normalization function to scale data between 0 and 1
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Compute range before normalization
range_bef <- apply(training[, -1], 2, range)

# Normalize the dataset
normalized_ds_train <- apply(training[, -1], 2, normalize)

# Compute range after normalization
range_af <- apply(normalized_ds_train, 2, range)

# Plot range before normalization
plot(range_bef, main = "Range Before Normalization", xlab = "Features", ylab = "Range")

# Plot range after normalization
plot(range_af, main = "Range After Normalization", xlab = "Features", ylab = "Range")


# Apply normalization function to all columns except the date column in the testing set
normalized_ds_test <- apply(testing[, -1], 2, normalize)

# Rename columns in the testing set to match the column names in the training set
colnames(normalized_ds_test) <- colnames(normalized_ds_train)


# Add the 18th and 19th hour attributes to the input vectors
# Define the input vectors as a list of character vectors
input_vectors_narx <- list(
  c("lag_1", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "lag_7", "hour_18", "hour_19"),
  c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "hour_18", "hour_19")
)

# Build NARX models
# Define an empty list to store the models
Narx_Models <- list()

# Use a for loop to iterate over the input vectors
for (i in 1:length(input_vectors_narx)) {
  # Build a MLP model using the build_neural_net function, passing in the normalized training and test datasets,
  Narx_Models[[i]] <- build_nn_model(normalized_ds_train, normalized_ds_test, input_vectors_narx[[i]], c(5))
}

normalized_ds_test <- as.data.frame(normalized_ds_test)

# Evaluate NARX models
# Define an empty list to store the evaluation metrics
eval_metrics_narx <- list()
# Use a for loop to iterate over the models
for (i in 1:length(Narx_Models)) {
  # Calculate the evaluation metrics (RMSE, MAE, MAPE, and sMAPE) for each model using the calculate_metrics function,
  # passing in the actual test set values and the predictions from the current model
  eval_metrics_narx[[i]] <- calculate_metrics(normalized_ds_test$hour_20, Narx_Models[[i]]$predictions)
}

# Create a comparison table for NARX models
# Create a data frame containing the Model_Description, RMSE, MAE, MAPE, and sMAPE columns
com_table_narx <- data.frame(
  Model_Description = c("NARX(1,10,11)", "NARX(2,10,11)", "NARX(3,10,11)", "NARX(3,6,10,11)", "NARX(4,6,10,11)"),
  RMSE = sapply(eval_metrics_narx, function(x) x$RMSE),
  MAE = sapply(eval_metrics_narx, function(x) x$MAE),
  MAPE = sapply(eval_metrics_narx, function(x) x$MAPE),
  sMAPE = sapply(eval_metrics_narx, function(x) x$sMAPE)
)
# Print the comparison table to the console
print(com_table_narx)


metrics_evaluation <- list()

for (i in 1:length(Narx_Models)) {
  metrics_evaluation[[i]] <- calculate_metrics(normalized_ds_test$hour_20, Narx_Models[[i]]$predictions)
}

# Add more models with different hidden layer structures and input vectors to create 12-15 models in total

# Efficiency comparison between one-hidden layer and two-hidden layer networks

# Build a one-hidden layer neural network
hidden_1_model <- build_nn_model(normalized_ds_train, normalized_ds_test, c("lag_1", "hour_18", "hour_19"), c(5))

# Build a two-hidden layer neural network
hidden_2_model <- build_nn_model(normalized_ds_train, normalized_ds_test, c("lag_1", "lag_2", "lag_3", "lag_4", "lag_7", "hour_18", "hour_19"), c(3, 2))

# Check the total number of weight parameters per network
hidden_1_num_weights <- sum(sapply(hidden_1_model$model$weights, length))
hidden_2_num_weights <- sum(sapply(hidden_2_model$model$weights, length))

# Print the number of weight parameters for each network
cat("Total number of weight parameters for the one-hidden layer network:", hidden_1_num_weights, "\n")
cat("Total number of weight parameters for the two-hidden layer network:", hidden_2_num_weights, "\n")


# Find the index of the best model based on the RMSE evaluation metric
best_model_index <- which.min(sapply(metrics_evaluation, function(x) x$RMSE))

# Get the best model and its predictions
# check the length of models list
length(Narx_Models)

# set best_model_index to a valid index
index_of_best_model <- 1

# get the best model
bestModel <- Narx_Models[[index_of_best_model]]
predictions_of_best_model_predictions <- bestModel$predictions

# Find the minimum and maximum values of the 'hour_20' variable in the training set
minimum_value <- min(training$hour_20)
maximus_value <- max(training$hour_20)

# Denormalize the model predictions using the min and max values of the 'hour_20' variable
predictions_denormalized <- denormalize(predictions_of_best_model_predictions, minimum_value, maximus_value)

# Plot the predicted output vs. desired output using a line chart
plot(testing$hour_20, type = "l", col = "red", xlab = "Time", ylab = "Hour 20.00 Consumption", main = "Line Chart of Desired VS  Predicted Output")
lines(predictions_denormalized, col = "black")
legend("topleft", legend = c("Desired Output", "Predicted Output"), col = c("red", "black"), lty=1, cex=0.8)