# -*- coding: utf-8 -*-
"""Untitled0.ipynb
"""

import pandas as pd
from google.colab import drive
from google.colab import data_table
import matplotlib.pyplot as plt
data_table.enable_dataframe_formatter()

drive.mount('/content/drive')
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/titanic.csv", sep = ";")

data.head(10)

data.head()
data.describe()
data.info()
data.value_counts()

fig, ax = plt.subplots(figsize=(16, 12))
data.hist(ax=ax)
plt.show()

data['Age'].fillna(data['Age'].median(), inplace=True)

data.rename(columns={'2urvived': 'Survived'}, inplace=True)

data.drop_duplicates(inplace=True)

data = data[['Age', 'Fare', 'Sex', 'Pclass', 'Survived']]

data.head()
data.describe()

fig, ax = plt.subplots(figsize=(16, 12))
data.hist(ax=ax)
plt.show()

import seaborn as sns 

sns.countplot(x='Survived', data=data)
plt.show()

sns.countplot(x='Survived', hue='Sex', data=data)
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=data)
plt.show()

g = sns.FacetGrid(data, col='Survived')
g.map(sns.histplot, 'Age')
plt.show()

sns.boxplot(x='Survived', y='Fare', data=data)
plt.show()

print(data.isnull().values.any())

corr_matrix = data.corr()
corr_matrix["Survived"].sort_values(ascending=False)

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

train_set = train_set.copy()
test_set = test_set.copy()

data = train_set.drop("Survived", axis=1)
data_label = train_set[["Survived"]].copy()
data_label_test = test_set[["Survived"]].copy()

data.head()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Define the column transformer for preprocessing the data
preprocessing = ColumnTransformer([
    ("normal", StandardScaler(), ["Age", "Fare"]),
    ("cat", OneHotEncoder(), ["Sex", "Pclass"])
])

# Fit and transform the data using the defined preprocessing steps
X = preprocessing.fit_transform(data)

print(data.dtypes)

data_prepared = preprocessing.fit_transform(data)
data_prepared.shape

#features for prediction = (17290, 31)

from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(data, data_label)

data_predictions = lin_reg.predict(data)

print(data_predictions)

#Train a linear regression model (LinearRegression) on data
#Features are preprocessed using preprocessing (a ColumnTransformer object)
#Target variable is preprocessed using preprocessing_label (another ColumnTransformer object)
#Model is trained on preprocessed data
#Predictions are made on data and stored in data_predictions

import numpy as np

data_predictions_orig = np.exp(data_predictions)
data_label_transformed_orig = np.exp(data_label_transformed)

abs_error = np.abs(data_predictions_orig - data_label_transformed_orig)
median_abs_error = np.median(abs_error)
rmse = np.sqrt(np.mean(np.square(data_predictions_orig - data_label_transformed_orig)))

median_price = np.median(data_label_transformed_orig)
pct_error = median_abs_error / median_price * 100

print(f"Median absolute error: {median_abs_error}")
print(f"RMSE: {rmse}")
print(f"Percentage error: {pct_error}")

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

data_label_transformed = data_label

forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_reg.fit(data, np.ravel(data_label_transformed))

def adjusted_rsquare(y_true, y_pred, **kwargs):
  return -np.sqrt(np.mean((np.exp(y_true)-np.exp(y_pred))**2))

neg_exp_root_mean_squared_error = make_scorer(adjusted_rsquare, greater_is_better=False)

forest_rmses = -cross_val_score(forest_reg, data, np.ravel(data_label_transformed),
 scoring=neg_exp_root_mean_squared_error, cv=10)

predicted_labels = forest_reg.predict(data)

rmse = np.sqrt(mean_squared_error(data_label_transformed, predicted_labels))

print("RMSE:", rmse)

"""random forest appears to work way better than the linear regression model """
