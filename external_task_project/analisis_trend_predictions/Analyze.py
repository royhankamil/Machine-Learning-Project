import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Get data
data_path = os.path.abspath(__file__) + '/../Data/aol_data.csv'
data = pd.read_csv(data_path)

production_data = data.iloc[0].tolist()

x_data = np.arange(1, 145)# months 1-145
y_data = np.array(production_data)

#model function
def polynomial_model(x, *coeffs):
    return np.polyval(coeffs, x)

# fit values
degree = 3
polynomial_coefficients = np.polyfit(x_data, y_data, degree)

polynomial_model_func = np.poly1d(polynomial_coefficients)

y_fitted_poly = polynomial_model_func(x_data)

#visualize scatted data with prediction
plt.figure(figsize=(14, 7))
plt.plot(x_data, y_data, label='Original Data')
plt.plot(x_data, y_fitted_poly, label=f'Predicted Line', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Number of Bags Produced')
plt.title('Polynomial Model Fit to Monthly Production of Bags (Jan 2018 - Dec 2023)')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the accuracy
residuals_poly = y_data - y_fitted_poly
ss_res_poly = np.sum(residuals_poly**2)
ss_tot_poly = np.sum((y_data - np.mean(y_data))**2)
r_squared_poly = 1 - (ss_res_poly / ss_tot_poly)

print('Polynomial Coefficients:', polynomial_coefficients)
print('R-squared (accuracy):', r_squared_poly)

a3, a2, a1, a0 = polynomial_coefficients # extract model
print(f'numerical form: y = {a3}x^3 + {a2}x^2 + {a1}x + {a0}')

# predict by mohnts
extended_months = np.arange(1, 201)
predicted_production = polynomial_model_func(extended_months)

capacity_limit = 25000
month_exceeds_capacity = np.argmax(predicted_production > capacity_limit) + 1

start_construction_month = month_exceeds_capacity - 13

# Ensure within valid range
if start_construction_month < 1:
    start_construction_month = 1

print("\nThe warehouse can store a maximum of 25,000 bags each month.")
print("The new warehouse construction will take 13 months")
print('Month when production exceeds warehouse capacity:', month_exceeds_capacity)
print('Month to start building new warehouse:', start_construction_month)


# extend predicted line
x_predict_ = np.arange(1, 190)
y_predict_ = polynomial_model_func(x_predict_)


# visualize with new data
plt.figure(figsize=(14, 7))
plt.scatter(x_data, y_data, label='Original Data', color='blue') # original data
plt.plot(x_predict_, y_predict_, label=f'Predicted Line', linestyle='--', color='red') # predicted data

plt.axhline(y=capacity_limit, color='gray', linestyle='--', label='Warehouse Capacity (25,000 bags)')
plt.axvline(x=month_exceeds_capacity, color='purple', linestyle='--', label=f'Month Exceeds Capacity: {month_exceeds_capacity}')
plt.axvline(x=start_construction_month, color='orange', linestyle='--', label=f'Start Construction Month: {start_construction_month}')

plt.xlabel('Month')
plt.ylabel('Number of Bags Produced')
plt.title('Polynomial Model Fit to Monthly Production of Bags (Jan 2018 - Dec 2023) and Future Predictions')
plt.legend()
plt.grid(True)
plt.show()