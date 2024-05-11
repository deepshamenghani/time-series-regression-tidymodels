# Load required libraries
library(tidyverse)
library(lubridate)
library(tidymodels)
library(modeltime)
library(viridis)
library(timetk)

# Load the data from a CSV file.
data <- read.csv("data/timeseries_data_single_storeproduct.csv")

data |> 
 glimpse()

data |> 
  summary()

data <- data |> 
  mutate(date = as.Date(date, format = "%m/%d/%Y"))

# Preprocess data to include time-related factors and handle missing values.
df <- data |> 
  mutate(
    year = year(date),  # Extract year from date
    semester = factor(semester(date)),  # Extract semester (1st or 2nd)
    quarter = factor(quarter(date)),  # Extract quarter (1 to 4)
    day_in_week = factor(wday(date, label = TRUE)),  # Day of the week as factor
    week_in_year = factor(week(date)),  # Week of the year
    day_in_year = factor(yday(date)),  # Day of the year
    month = factor(month(date, label = TRUE))  # Month as a factor
  )

glimpse(df)  # Quick view of the data structure to confirm transformations

# Data Visualization
## Line plot of sales consumption over time and color by weekend
df %>%
  ggplot(aes(date, sales)) +
  geom_line(alpha = 1, size = 1, color = "darkblue") +  
  theme_bw() +
  labs(title = "Daily Sales Distribution Over Time", x = "Date", y = "Sales") +
  scale_x_date(date_labels = "%Y", date_breaks = "2 years") +  # Format the date on the x-axis
  scale_y_continuous(labels = scales::comma)  # Format y-axis with commas for thousands

## Boxplot of sales consumption by day of the week
df |>
  ggplot(aes(day_in_week, sales, color = day_in_week)) +
  geom_boxplot() +
  geom_jitter(alpha = 0.1) +
  theme_bw() +
  scale_colour_viridis_d() +
  labs(title = "Daily Sales Distribution by Day of Week", x = "Day of the Week", y = "Sales") +
  scale_x_discrete() +
  scale_y_continuous(labels = scales::comma)

## Violin and jitter plot of sales consumption by month
df |> 
  ggplot(aes(month, sales)) +
  geom_violin(color = "darkgreen") +  
  geom_jitter(alpha = 0.2, aes(color = sales)) +  
  theme_light() +
  geom_smooth(method = "loess", se = FALSE) +  
  scale_colour_viridis_c() +
  labs(title = "Daily Sales Distribution by Month", x = "", y = "Sales", color= "Sales") 
  
# Data Splitting for Model Training
## Split the data into training and testing sets to prepare for model training and validation.

df_split <- df |> 
  initial_time_split(prop=0.9)  # 90% of data for training, 10% for testing

df_train <- training(df_split)
df_test <- testing(df_split)

# Combining the datasets
df_combined <- bind_rows(df_train |> mutate(set = "Training"), df_test |> mutate(set = "Testing"))

# Plotting the train/test split
ggplot(df_combined, aes(x = date, y = sales, color = set)) +
  geom_line() +  # Use geom_point if your data is better represented by points
  scale_color_manual(values = c("Training" = "#1f77b4", "Testing" = "#ff7f0e")) +  # Custom color selection
  labs(
    title = "Visualization of Train/Test Data Split",
    x = "Date",
    y = "Sales",
    color = "Dataset"
  ) +
  theme_minimal() +
  scale_x_date(date_labels = "%Y", date_breaks = "2 years") +
  theme(legend.position = "top")

set.seed(123)  # Set seed for reproducibility

# Creating cross-validation folds of time series data
df_folds <- time_series_cv(df_train, initial = "3 years", assess = "1 year", skip = "6 months", slice_limit = 5)  # Time-series cross-validation setup

plot_time_series_cv_plan(df_folds, date, sales)

# Data Preparation for Modeling
## Create a recipe object for preprocessing the training data to be used in model fitting.

# Recipe for auto arima boost model
recipe_autoarima <- 
  recipe(sales ~ date,
         data = df_train)

# Recipe for rf
recipe_rf <- 
  recipe(sales ~ ., data = df_train) |>
  step_holiday(date, holidays = timeDate::listHolidays("US")) |>  # Add binary indicators for public holidays in France
  step_rm(date) |>  # Remove the date column to prevent data leakage
  step_dummy(all_nominal_predictors())

recipe_rf |> prep() |> bake(new_data = NULL)

# Model Setup and Resampling
## Define the specifications for various regression models including arima, random forest, and boosted trees.

auto_arima_spec <- arima_boost() %>% 
  set_mode("regression") |> 
  set_engine('auto_arima_xgboost')

rf_spec <- 
  rand_forest(trees = 500) |>  # Random forest with 500 trees
  set_mode("regression") |> 
  set_engine("ranger")

## Create workflow set for modeling and fitting across all three models

workflowset_df <- 
  workflow_set(
    list(recipe_autoarima, recipe_rf),
    list(auto_arima_spec, rf_spec),
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
evaluation_metrics <- metric_set(rmse, mae, rsq)
results <- evaluation_metrics(predictions, truth = sales, estimate = .pred) 
print(results)  # Print the evaluation results to assess model accuracy

