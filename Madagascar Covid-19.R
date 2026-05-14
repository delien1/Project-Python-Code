# Load required libraries
library(dplyr)
library(tidyr)
library(neuralnet)
library(ggplot2)
library(caret)

# Read the data
data <- read.csv("D:/Documents/Multivariate Statistics1/NNET in R/covid19 Dataset.csv")
View(data)
# 1. Extract Madagascar's daily COVID-19 cases
madagascar_data <- data %>%
  filter(Country.Region == "Madagascar") %>%
  select(Date, Confirmed) %>%
  arrange(Date) %>%
  group_by(Date) %>%
  summarise(Daily_Cases = sum(Confirmed)) %>%
  ungroup()

# Remove rows with all zeros at the beginning (no cases)
madagascar_data <- madagascar_data %>%
  filter(cumsum(Daily_Cases) > 0)

# Create time index
madagascar_data$t <- 1:nrow(madagascar_data)
View(madagascar_data)
# 2. Create lagged variables (x_{t-1}, x_{t-2}, x_{t-3}, x_{t-4}, x_{t-5})
madagascar_lagged <- madagascar_data %>%
  mutate(
    x_t_minus_1 = lag(Daily_Cases, 1),
    x_t_minus_2 = lag(Daily_Cases, 2),
    x_t_minus_3 = lag(Daily_Cases, 3),
    x_t_minus_4 = lag(Daily_Cases, 4),
    x_t_minus_5 = lag(Daily_Cases, 5)
  ) %>%
  filter(!is.na(x_t_minus_5))  # Remove rows with NA values due to lagging

# Display the table with lagged values
print("Madagascar COVID-19 Daily Cases with Lagged Variables:")
print(madagascar_lagged)

# 3. Fit ANN Model to predict x_t using lagged variables

# Prepare data for ANN (remove Date column and keep only numeric variables)
ann_data <- madagascar_lagged %>%
  select(-Date, -t)

# Check if we have enough data
if(nrow(ann_data) < 10) {
  stop("Not enough data points for ANN modeling. Need at least 10 observations.")
}

# Normalize the data for better ANN performance (scale to 0-1)
normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

denormalize <- function(x, min_val, max_val) {
  x * (max_val - min_val) + min_val
}

# Store original min and max for denormalization
min_cases <- min(ann_data$Daily_Cases)
max_cases <- max(ann_data$Daily_Cases)

# Normalize all columns
ann_data_normalized <- as.data.frame(lapply(ann_data, normalize))

# Split data into training and testing sets (80-20 split)
set.seed(123)
train_index <- createDataPartition(ann_data_normalized$Daily_Cases, p = 0.8, list = FALSE)
train_data <- ann_data_normalized[train_index, ]
test_data <- ann_data_normalized[-train_index, ]

# Define the ANN formula
ann_formula <- Daily_Cases ~ x_t_minus_1 + x_t_minus_2 + x_t_minus_3 + x_t_minus_4 + x_t_minus_5

# Train the ANN model with error handling
tryCatch({
  ann_model <- neuralnet(
    ann_formula,
    data = train_data,
    hidden = c(3, 2),  # Simpler architecture: two hidden layers with 3 and 2 neurons
    linear.output = TRUE,
    threshold = 0.1,    # Increased threshold for faster convergence
    stepmax = 1e5,      # Reduced stepmax
    lifesign = "full",  # Show training progress
    lifesign.step = 10
  )
  
  # Print the model summary only if model was created successfully
  print("ANN Model Summary:")
  print(ann_model)
  
  # Make predictions on test data
  predictions_normalized <- predict(ann_model, test_data)
  
  # Denormalize predictions
  predictions_denormalized <- denormalize(predictions_normalized, min_cases, max_cases)
  actual_values <- denormalize(test_data$Daily_Cases, min_cases, max_cases)
  
  # 4. Assess goodness of fit using R²
  ss_res <- sum((actual_values - predictions_denormalized)^2)
  ss_tot <- sum((actual_values - mean(actual_values))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  
  print(paste("R-squared (Goodness of Fit):", round(r_squared, 4)))
  
  # Calculate additional metrics
  mse <- mean((actual_values - predictions_denormalized)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(actual_values - predictions_denormalized))
  
  print(paste("Mean Squared Error (MSE):", round(mse, 4)))
  print(paste("Root Mean Squared Error (RMSE):", round(rmse, 4)))
  print(paste("Mean Absolute Error (MAE):", round(mae, 4)))
  
  # 5. Visualize x_t over time
  # Plot 1: Time series of actual daily cases
  p1 <- ggplot(madagascar_data, aes(x = t, y = Daily_Cases)) +
    geom_line(color = "blue", linewidth = 1) +
    geom_point(color = "red", size = 1) +
    labs(
      title = "Madagascar COVID-19 Daily Cases Over Time",
      x = "Time (t)",
      y = "Daily Cases (x_t)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.title = element_text(face = "bold")
    )
  
  print(p1)
  show(plot)
  # Plot 2: Actual vs Predicted values (if we have test predictions)
  if(length(actual_values) > 0 && length(predictions_denormalized) > 0) {
    comparison_df <- data.frame(
      Time = 1:length(actual_values),
      Actual = actual_values,
      Predicted = predictions_denormalized
    )
    
    p2 <- ggplot(comparison_df, aes(x = Time)) +
      geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
      geom_line(aes(y = Predicted, color = "Predicted"), linewidth = 1, linetype = "dashed") +
      scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
      labs(
        title = "Actual vs Predicted Daily Cases",
        x = "Time Index",
        y = "Daily Cases",
        color = "Legend"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.title = element_text(face = "bold"),
        legend.position = "bottom"
      )
    
    print(p2)
    show(plot)
    # Plot 3: Scatter plot of Actual vs Predicted
    p3 <- ggplot(comparison_df, aes(x = Actual, y = Predicted)) +
      geom_point(color = "darkgreen", alpha = 0.7) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
      labs(
        title = "Actual vs Predicted Values",
        subtitle = paste("R² =", round(r_squared, 4)),
        x = "Actual Daily Cases",
        y = "Predicted Daily Cases"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        axis.title = element_text(face = "bold")
      )
    
    print(p3)
  }
  
  # Display model structure
  tryCatch({
    plot(ann_model, rep = "best")
  }, error = function(e) {
    print("Could not plot neural network structure")
  })
  
}, error = function(e) {
  print(paste("Error in ANN training:", e$message))
  print("Trying alternative approach with simpler model...")
  
  # Alternative: Use linear model as fallback
  lm_model <- lm(ann_formula, data = train_data)
  print("Linear Model Summary (Fallback):")
  print(summary(lm_model))
  
  # Make predictions
  predictions_normalized <- predict(lm_model, test_data)
  predictions_denormalized <- denormalize(predictions_normalized, min_cases, max_cases)
  actual_values <- denormalize(test_data$Daily_Cases, min_cases, max_cases)
  
  # Calculate R-squared
  ss_res <- sum((actual_values - predictions_denormalized)^2)
  ss_tot <- sum((actual_values - mean(actual_values))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  
  print(paste("Linear Model R-squared:", round(r_squared, 4)))
})

# Print final data summary
cat("\n=== DATA SUMMARY ===\n")
cat("Total observations:", nrow(madagascar_data), "\n")
cat("Observations after lagging:", nrow(madagascar_lagged), "\n")
cat("Date range:", min(madagascar_data$Date), "to", max(madagascar_data$Date), "\n")
cat("Daily cases range:", min(madagascar_data$Daily_Cases), "to", max(madagascar_data$Daily_Cases), "\n")