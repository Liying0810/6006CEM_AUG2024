# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 23:08:36 2024

@author: Keoh Li Ying
"""

# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Step 2: Load the dataset (Kaggle's House Prices dataset)
house_prices_data = pd.read_csv('C:/Users/adeli/Downloads/house_prices_dataset.csv')

# Step 3: Define features and target variable
X = house_prices_data[['BedroomAbvGr', 'GrLivArea', 'Neighborhood']]  # Example features
X = pd.get_dummies(X)  # One-hot encode categorical features
y = house_prices_data['SalePrice']

# Step 4: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Step 6: Train Decision Tree Regressor model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Step 7: Make predictions on the validation set
y_pred_linear = linear_model.predict(X_val)
y_pred_tree = tree_model.predict(X_val)

# Step 8: Evaluate the models using Mean Squared Error (MSE)
mse_linear = mean_squared_error(y_val, y_pred_linear)
mse_tree = mean_squared_error(y_val, y_pred_tree)

print(f"Linear Regression MSE: {mse_linear}")
print(f"Decision Tree MSE: {mse_tree}")

# For Ridge Regression (regularization parameter alpha)
ridge = Ridge()
param_grid_ridge = {'alpha': [0.1, 1, 10, 100]}
grid_search_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='neg_mean_squared_error')
grid_search_ridge.fit(X_train, y_train)
best_ridge = grid_search_ridge.best_estimator_
y_pred_ridge = best_ridge.predict(X_val)
ridge_mse = mean_squared_error(y_val, y_pred_ridge)

# For Decision Tree (tuning max_depth)
param_grid_tree = {'max_depth': [5, 10, 15, 20]}
grid_search_tree = GridSearchCV(tree_model, param_grid_tree, cv=5, scoring='neg_mean_squared_error')
grid_search_tree.fit(X_train, y_train)
best_tree = grid_search_tree.best_estimator_
y_pred_tree_tuned = best_tree.predict(X_val)
tree_mse_tuned = mean_squared_error(y_val, y_pred_tree_tuned)

print(f"Ridge Regression MSE: {ridge_mse}")
print(f"Tuned Decision Tree MSE: {tree_mse_tuned}")


