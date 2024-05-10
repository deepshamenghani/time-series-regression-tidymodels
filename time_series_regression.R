# Load required libraries
library(tidyverse)
library(lubridate)
library(tidymodels)
library(modeltime)
library(viridis)
library(timetk)

# Load the data from a CSV file.
data <- read.csv("data/daily_energy_comsumption.csv")

# Preprocess data to include time-related factors and handle missing values.
data_date_features <- data |> 
  mutate(date = as.Date(date, format = "%m/%d/%Y")) |>  # Convert string Date to Date object
  mutate(
    year = year(date),  # Extract year from date
    semester = factor(semester(date)),  # Extract semester (1st or 2nd)
    quarter = factor(quarter(date)),  # Extract quarter (1 to 4)
    day_in_week = factor(wday(date, label = TRUE)),  # Day of the week as factor
    week_in_year = factor(week(date)),  # Week of the year
    day_in_year = factor(yday(date)),  # Day of the year
    month = factor(month(date, label = TRUE))  # Month as a factor
  )

# Handle missing values by imputing closest value from the same day of week
df <- data_date_features |> 
  group_by(day_in_week) |> 
  fill(energy, .direction = "downup") |>  # Fill NA values by carrying the last observation forward and backward
  ungroup()

glimpse(df)  # Quick view of the data structure to confirm transformations

# Data Visualization
## Line plot of energy consumption over time and color by weekend
df |> 
  mutate(
    weekend = as.factor(case_when(
      day_in_week %in% c("Sat", "Sun") ~ "Weekend",  # Create a weekend identifier
      TRUE ~ "Weekday"
    ))
  ) |>
  ggplot(aes(date, energy, color = weekend)) +
  geom_line(alpha = 0.5, size = 1) +  # Line plot with transparency and size settings
  theme_minimal() +
  labs(title = "Daily Energy distribution by weekend")

## Boxplot of energy consumption by day of the week
df |> 
  ggplot(aes(day_in_week, energy, color = day_in_week)) +
  geom_boxplot() +  # Create boxplots grouped by day of the week
  geom_jitter(alpha = 0.1) +  # Add jitter to show individual data points
  theme_minimal() +
  scale_colour_viridis_d() +
  labs(title = "Daily Energy distribution by day of week")

## Scatter plot of energy consumption by day of the year with weekend vs weekday trend line
df |> 
  mutate(
    weekend = as.factor(case_when(
      day_in_week %in% c("Sat", "Sun") ~ "Weekend",
      TRUE ~ "Weekday"
    ))
  ) |>
  ggplot(aes(as.numeric(day_in_year), energy, color = weekend)) +
  geom_point(alpha = 0.1, size = 2) +  # Point plot for daily data
  theme_minimal() +
  geom_smooth(method = "loess", se = FALSE) +  # Add LOESS curve for trend
  labs(title = "Daily Energy distribution by day of year")

## Violin and jitter plot of energy consumption by month
df |> 
  ggplot(aes(month, energy)) +
  geom_violin(color = "darkgreen") +  # Create violin plots for density estimation
  geom_jitter(alpha = 0.2, aes(color = energy)) +  # Overlay jittered points to show raw data
  theme_minimal() +
  geom_smooth(method = "loess", se = FALSE) +  # Smooth curve to show trend
  scale_colour_viridis_c() +
  labs(title = "Daily Energy distribution by month")

# Data Splitting for Model Training
## Split the data into training and testing sets to prepare for model training and validation.

df_split <- df |> 
  initial_time_split(prop=0.9)  # 90% of data for training, 10% for testing

df_train <- training(df_split)
df_test <- testing(df_split)

set.seed(123)  # Set seed for reproducibility

# Creating cross-validation folds of time series data
df_folds <- time_series_cv(df_train, initial = "2 years", assess = "3 months", slice_limit = 10)  # Time-series cross-validation setup

# Data Preparation for Modeling
## Create a recipe object for preprocessing the training data to be used in model fitting.

# Recipe for auto arima boost model
recipe_autoarima <- 
  recipe(energy ~ date,
         data = df_train)

# Recipe for xgboost
recipe_rf <- 
  recipe(energy ~ ., data = df_train) |>
  step_holiday(date, holidays = timeDate::listHolidays("FR")) |>  # Add binary indicators for public holidays in France
  step_rm(date) |>  # Remove the date column to prevent data leakage
  step_dummy(all_nominal_predictors())  # Convert all nominal variables into dummy variables for modeling


# Recipe for xgboost
recipe_xgb <- 
  recipe(energy ~ ., data = df_train) |>
  step_holiday(date, holidays = timeDate::listHolidays("FR")) |>  # Add binary indicators for public holidays in France
  step_rm(date) |>  # Remove the date column to prevent data leakage
  step_dummy(all_nominal_predictors())  # Convert all nominal variables into dummy variables for modeling


# Model Setup and Resampling
## Define the specifications for various regression models including arima, random forest, and boosted trees.

auto_arima_spec <- arima_boost() %>% 
  set_mode("regression") |> 
  set_engine('auto_arima_xgboost')

rf_spec <- 
  rand_forest(trees = 500) |>  # Random forest with 500 trees
  set_mode("regression") |> 
  set_engine("ranger")

xgb_spec <- 
  boost_tree(trees = 500) |>  # Boosted trees with 500 iterations
  set_mode("regression") |> 
  set_engine("xgboost")

## Create workflow set for modeling and fitting across all three models

workflowset_df <- 
  workflow_set(
    list(recipe_autoarima, recipe_rf, recipe_xgb),
    list(auto_arima_spec, rf_spec, xgb_spec),
    cross=FALSE
  )


## Fit models using the defined training data and re sampling plan, and collect the performance metrics.

df_results <-
  workflow_map(
    workflowset_df,
    "fit_resamples",
    resamples = df_folds
  )

autoplot(df_results)  # Visualize the re-sampling results to compare model performances

# Selecting Best Model Based on RMSE
## Analyze the model results to select the best performing model based on the Root Mean Square Error (RMSE).
best_workflow_id <- df_results %>%
  rank_results(rank_metric = "rmse") %>%
  head(1) %>%
  pull(wflow_id)  # Identify and extract the ID of the best model based on lowest RMSE

## Retrieve the best parameters from the workflow with the highest performance score.
best_params <- df_results %>%
  extract_workflow_set_result(id = best_workflow_id) %>%
  select_best(metric = "rmse")  # Select the best parameters for the model with the lowest RMSE

## Apply the best parameters to the selected model and prepare it for final evaluation.
best_workflow <- df_results %>%
  extract_workflow(id = best_workflow_id)  # Extract the workflow of the best model

finalized_workflow <- finalize_workflow(best_workflow, best_params)  # Finalize the workflow with the best parameters

# Fit the model to the training data
## Use the trained final model to make predictions on the test dataset.
predictions <- finalized_workflow %>%
  fit(df_train) %>%
  augment(df_test)  

## Calculate and review evaluation metrics to assess the performance of the final model.
evaluation_metrics <- metric_set(rmse)
results <- evaluation_metrics(predictions, truth = energy, estimate = .pred) 
print(results)  # Print the evaluation results to assess model accuracy

## Conclude with the selection of the best model based on the evaluation and summarize its performance.
selected_model <- best_workflow_id  # Specify the model type that was selected
selected_model_rmse <- results$.estimate[results$.metric == "rmse"]  # Retrieve the RMSE value for the selected model
