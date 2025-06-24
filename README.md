# Housing Price Prediction: Regression and Pipeline Modeling

## Overview

This project is the final assignment for the **IBM Data Analysis with Python** certification course on **Coursera**. It demonstrates key steps in the data science workflow including data cleaning, exploratory data analysis (EDA), regression modeling (linear, polynomial, and ridge), and pipeline integration. The analysis is performed on a real-world housing dataset.

Developed by **Mgr. Stefan Vach** as part of the **IBM Data Analyst Professional Certificate**.

## Objectives

- Clean and preprocess a real-world dataset with missing values.
- Conduct exploratory data analysis using visualizations.
- Apply simple and multiple linear regression models.
- Enhance prediction performance using polynomial features.
- Implement regularization with Ridge regression.
- Build reusable pipelines combining multiple preprocessing and modeling steps.

## Dataset

- **Source**: [IBM Developer Skills Network – Coursera](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv)
- **Content**: Real estate transaction records for King County, USA (includes prices, square footage, number of bedrooms, waterfront presence, and more).

## Tools and Libraries

- **Python 3**
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Matplotlib & Seaborn**: Visualization
- **scikit-learn**: Modeling (Linear Regression, Polynomial Features, Ridge, Pipeline)

## Key Techniques

### Data Preprocessing
- Removal of irrelevant features (`id`, `Unnamed: 0`)
- Handling missing values in `bedrooms` and `bathrooms` using mean imputation

### Exploratory Data Analysis (EDA)
- Boxplots (e.g., `price` vs `waterfront`)
- Scatterplots with regression lines (e.g., `sqft_above` vs `price`)
- Correlation analysis using a heatmap

### Regression Modeling
- **Simple Linear Regression**: `sqft_living` → `price`
- **Multiple Linear Regression**: 11 predictors including `bathrooms`, `grade`, `sqft_basement`
- **Polynomial Regression** with `StandardScaler`
- **Ridge Regression**: Regularized model to prevent overfitting
- **Pipeline**: Combines preprocessing (scaling, polynomial expansion) and modeling

## Evaluation Metric

- **R² Score (Coefficient of Determination)** used to evaluate model performance on test data.

## Sample Output

