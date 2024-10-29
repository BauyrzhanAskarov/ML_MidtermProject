import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import mean_squared_error
from scipy.linalg import pinv
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('insurance.csv')
X = data['bmi'].values.reshape(-1, 1)
y = data['charges'].values

# Prepare the design matrix by adding a bias term
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Calculate weights using the pseudo-inverse formula
start_time = time()
theta_optimal = pinv(X_b).dot(y)
computation_time = time() - start_time

print("Optimal weights:", theta_optimal)
print("Computation Time (seconds):", computation_time)

# Make predictions and plot the regression line
y_pred = X_b.dot(theta_optimal)

plt.plot(X, y, "b.", label="Data points")
plt.plot(X, y_pred, "r-", label="Regression line")
plt.xlabel("BMI")
plt.ylabel("Insurance Charges")
plt.legend()
plt.title("Linear Regression using Pseudo-Inverse")
plt.show()

# Evaluate with Mean Squared Error
mse_custom = mean_squared_error(y, y_pred)
print("Mean Squared Error (Custom Implementation):", mse_custom)

# Comparison with Scikit-Learn's Linear Regression
start_time_sklearn = time()
lin_reg = LinearRegression()
lin_reg.fit(X, y)
computation_time_sklearn = time() - start_time_sklearn
y_pred_sklearn = lin_reg.predict(X)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)

print("\nScikit-Learn Linear Regression")
print("Optimal Weights (Intercept and Slope):", np.r_[lin_reg.intercept_, lin_reg.coef_])
print("Mean Squared Error (Scikit-Learn):", mse_sklearn)
print("Computation Time (seconds):", computation_time_sklearn)

# Comparison of Implementations
print("\nComparison of Implementations:")
print("Difference in Mean Squared Error:", abs(mse_custom - mse_sklearn))
print("Difference in Computation Time:", abs(computation_time - computation_time_sklearn))
