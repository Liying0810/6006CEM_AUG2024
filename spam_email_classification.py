# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 07:13:46 2024

@author: Keoh Li Ying
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('C:/Users/adeli/Downloads/spam_email_dataset.csv')

# Check for missing values and clean if necessary
df.dropna(inplace=True)

# Split the dataset into features (X) and labels (y)
X = df['text']  # Features: email content
y = df['spam']  # Labels: 1 for spam, 0 for not spam

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)  # Apply TF-IDF to X_train
X_test_tfidf = tfidf.transform(X_test)  # Transform X_test based on the same TF-IDF fitting

# Apply SMOTE to balance classes in the training set (on the transformed TF-IDF data)
smote = SMOTE(random_state=42)
X_train_tfidf_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

# Logistic Regression Model with class balancing
log_reg = LogisticRegression(class_weight='balanced')
log_reg.fit(X_train_tfidf_res, y_train_res)  # Use resampled data for training
y_pred_log_reg = log_reg.predict(X_test_tfidf)

# Logistic Regression Metrics
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print(f'Logistic Regression Accuracy: {log_reg_accuracy * 100:.2f}%')

# Classification Report for Logistic Regression
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

# AUC and ROC Curve for Logistic Regression
y_pred_prob_log_reg = log_reg.predict_proba(X_test_tfidf)[:, 1]
log_reg_auc = roc_auc_score(y_test, y_pred_prob_log_reg)
print(f'Logistic Regression AUC: {log_reg_auc:.2f}')

# ROC Curve
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_prob_log_reg)
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {log_reg_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Random Forest Classifier Model with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
rf_grid_search.fit(X_train_tfidf_res, y_train_res)  # Use resampled data for training
best_rf = rf_grid_search.best_estimator_

# Evaluate Random Forest with best parameters
y_pred_rf = best_rf.predict(X_test_tfidf)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')

# Classification Report for Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# AUC and ROC Curve for Random Forest
y_pred_prob_rf = best_rf.predict_proba(X_test_tfidf)[:, 1]
rf_auc = roc_auc_score(y_test, y_pred_prob_rf)
print(f'Random Forest AUC: {rf_auc:.2f}')

# Plot ROC Curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
