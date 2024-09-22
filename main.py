import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load and inspect the dataset
data = pd.read_csv('credit_risk_dataset.csv')

# Preview the dataset
print(data.head())
print(data.info())  # Check data types and missing values
print(data.describe())  # Summary statistics

# Print column names to find the correct target column
print("Column names:", data.columns)

# Fill missing values only in numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Identify the correct target column name
target_column_name = 'loan_status'  # Update to the actual target column name

# Check if the target column exists
if target_column_name not in data.columns:
    print("Column names:", data.columns)
    raise KeyError(f"'{target_column_name}' not found in dataset. Please check the correct target column name.")

# Separate features (X) and target variable (y)
X = data.drop(target_column_name, axis=1)  # Features
y = data[target_column_name]  # Target variable


# Feature scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence issues occur
model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions with Logistic Regression
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-validation for Logistic Regression
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
print("Cross-Validation Accuracy: ", cv_scores.mean())

# Hyperparameter tuning for Logistic Regression using GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best Hyperparameters:", grid.best_params_)

# Save the trained model using joblib
joblib.dump(model, 'credit_risk_model.pkl')

# Confusion matrix visualization using heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Feature importance visualization for Random Forest
feature_importances = rf_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

plt.barh(range(X_train.shape[1]), feature_importances[sorted_indices])
plt.title('Feature Importance - Random Forest')
plt.show()
