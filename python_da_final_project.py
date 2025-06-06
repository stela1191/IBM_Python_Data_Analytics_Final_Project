"""
Final Project Code – IBM Data Analysis with Python Certification

This notebook demonstrates key data analysis and regression modeling techniques using Python.
It includes data preprocessing, exploratory data analysis (EDA), model building with linear and polynomial regression,
and performance evaluation using real housing data. The final steps apply Ridge regression and a pipeline for
robust prediction.

Note: Due to limitations in JupyterLite, downloading the notebook wasn't possible, so here is the full raw code.
"""

# ------------------------------
# Suppress Warnings (for cleaner output)
# ------------------------------
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# ------------------------------
# Imports
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Enable inline plotting
%matplotlib inline

# ------------------------------
# Download and Load Dataset
# ------------------------------
import piplite
await piplite.install('seaborn')  # Required in JupyterLite
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
await download(filepath, "housing.csv")

df = pd.read_csv("housing.csv")

# ------------------------------
# Initial Exploration
# ------------------------------
print(df.head())
print(df.dtypes)
print(df.describe())

# Drop unneeded columns
df.drop(["id", "Unnamed: 0"], axis=1, inplace=True)
print(df.describe())

# ------------------------------
# Handling Missing Values
# ------------------------------
print("NaNs in 'bedrooms':", df['bedrooms'].isnull().sum())
print("NaNs in 'bathrooms':", df['bathrooms'].isnull().sum())

df['bedrooms'].replace(np.nan, df['bedrooms'].mean(), inplace=True)
df['bathrooms'].replace(np.nan, df['bathrooms'].mean(), inplace=True)

print("NaNs after replacement – 'bedrooms':", df['bedrooms'].isnull().sum())
print("NaNs after replacement – 'bathrooms':", df['bathrooms'].isnull().sum())

# ------------------------------
# Value Counts for Floors
# ------------------------------
floor_counts = df['floors'].value_counts().to_frame()
print(floor_counts)

# ------------------------------
# Boxplot: Price by Waterfront View
# ------------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(x='waterfront', y='price', data=df)
plt.title('Price Distribution by Waterfront View')
plt.xlabel('Waterfront (0 = No, 1 = Yes)')
plt.ylabel('Price')
plt.show()

# ------------------------------
# Scatter Plot with Regression Line: sqft_above vs. price
# ------------------------------
plt.figure(figsize=(8, 6))
sns.regplot(x='sqft_above', y='price', data=df)
plt.title('Correlation between Sqft Above and Price')
plt.xlabel('Square Feet Above')
plt.ylabel('Price')
plt.show()

# ------------------------------
# Correlation Matrix
# ------------------------------
df_numeric = df.select_dtypes(include=[np.number])
print(df_numeric.corr()['price'].sort_values())

# ------------------------------
# Simple Linear Regression: sqft_living vs. price
# ------------------------------
X = df[['sqft_living']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R² score (simple linear regression):", r2)

# ------------------------------
# Multiple Linear Regression with Selected Features
# ------------------------------
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement",
            "view", "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

X = df[features]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R² score (multiple linear regression):", r2)

# ------------------------------
# Pipeline with Polynomial Features + Scaling + Linear Regression
# ------------------------------
estimators = [
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('model', LinearRegression())
]

pipe = Pipeline(estimators)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

r2 = r2_score(y_test, y_pred)
print("R² score (polynomial pipeline):", r2)

# ------------------------------
# Ridge Regression
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R² score (Ridge regression):", r2)

# ------------------------------
# Polynomial + Ridge Regression
# ------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = poly.fit_transform(X_train)
x_test_poly = poly.transform(X_test)

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(x_train_poly, y_train)

y_pred = ridge_model.predict(x_test_poly)
r2 = r2_score(y_test, y_pred)
print("R² score (polynomial Ridge regression):", r2)
