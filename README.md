# Time Series Regression and Cross-Validation with Tidymodels, Modeltime, and Timetk

This repository hosts the R scripts and datasets featured in the detailed blog post titled "Time Series Regression and Cross-Validation with Tidymodels, Modeltime, and Timetk." The primary focus of this tutorial is to demonstrate advanced techniques in time series forecasting, specifically targeting sales data using modern packages in R. This project serves as an educational tool for those interested in learning about the integration of machine learning techniques into time series analysis using R's Tidymodels ecosystem.

## Repository Structure

- **/data/**: Contains the datasets utilized in the analyses. These data represent simulated sales information over a decade, specifically designed for time series modeling. The data was sourced from [Kaggle](https://www.kaggle.com/datasets/samuelcortinhas/time-series-practice-dataset).
- **time_series_regression.R**: The main R script file where the time series models are built and evaluated.
- **time-series-regression-tidymodels.Rproj**: R Project file.
- **README.md**: This file.
- **.gitignore**: Specifies intentionally untracked files to ignore.
- **.Rhistory**: R history file.

## Getting Started

### Prerequisites

Ensure you have R installed on your machine. You can download it from [CRAN](https://cran.r-project.org/).

### Installation

You will need to install several R packages to run the scripts. Execute the following commands in R:

```R
install.packages("tidyverse")
install.packages("lubridate")
install.packages("tidymodels")
install.packages("modeltime")
install.packages("timetk")
install.packages("viridis")
```

## Running the Analysis

To replicate the analysis:

1. Clone or download this GitHub repository to your local machine.
2. Open `time-series-regression-tidymodels.Rproj` with RStudio to set the correct working directory and environment.
3. Execute the script `time_series_regression.R` to perform the data analysis and modeling. This script processes the data, builds models, and evaluates their performance.
