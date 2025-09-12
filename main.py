import numpy as np
import random
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("data.csv")

# Objective function using mean squared error
def mse(act_val, pred_val):
    error = act_val - pred_val
    mse = np.mean(error**2)
    return mse

# Neighbor function to pick new solution
def neighbor(current_solution, step_size=0.1):
    # Copy current solution to avoid changing it directly
    new_solution = current_solution.copy()

    # Pick a random index
    ind = np.random.randint(len(current_solution))

    # Add a small change
    new_solution[ind] += np.random.uniform(-step_size, step_size)

    return new_solution

# Returns sum of all rows after weight has been applied 
def predict(weights, data):
    return np.dot(data, weights)

def sim_ann(data, target_vals, bounds, n_iterations=800, step_size=0.1, temp=10):
    # Assign random starting weights
    current_solution = [random.uniform(bound[0], bound[1]) for bound in bounds]
    current_eval = mse(target_vals, predict(current_solution, data))
    best_solution = current_solution
    best_eval = current_eval

    for i in range(n_iterations):
        potential = neighbor(current_solution)
        potential_eval = mse(target_vals, predict(potential, data))

        # Accept or reject potential weights for new current solution
        if potential_eval < current_eval or random.random() < math.exp((current_eval - potential_eval)/temp):
            current_solution = potential
            current_eval = potential_eval

        # Updates best solution if current solution is better
        if potential_eval < best_eval:
            best_solution = potential
            best_eval = potential_eval

        # Reduces temp(cool down)
        temp *= .99

    return best_solution, best_eval

def gradient_descent(data, target_vals, weights, lr=0.01, n_iter=100, alpha_l1=0.1, alpha_l2=0.1):
    for _ in range(n_iter):
        pred = predict(weights, data)
        error = pred - target_vals

        # Calculate the gradient for each weight:
        # 1. Transpose the data so that each feature is aligned for summing across all samples.
        # 2. Multiply the transposed data by the prediction errors to see how each feature contributes to the error.
        # 3. Divide by the number of samples to get the average contribution.
        # 4. Multiply by 2 because the derivative of the squared error includes a factor of 2.
        gradient = 2 * np.dot(data.T, error) / len(target_vals)
    
        # Add L2 penalty for Ridge
        gradient += 2 * alpha_l2 * weights

        # Update weights
        weights = weights - lr * gradient

        # Add L1 penalty for Lasso
        weights = np.sign(weights) * np.maximum(np.abs(weights) - lr * alpha_l1, 0.0)

    return weights, mse(target_vals, predict(weights, data))

# Select which rows from the data table will be used
air_qual_feat = ["pm2.5", "no2", "co2"]
health_risk_feat = ["airQuality", "tempmax", "tempmin", "temp", "dew", "humidity", "precip", "precipprob", 
"precipcover", "windgust", "pressure", "cloudcover", "visibility", "solarradiation", "solarenergy", "uvindex",
"severerisk", "moonphase", "datetimeEpoch"]

# Create a bound array for each feature
air_qual_bounds = [(-1, 1)] * len(air_qual_feat)
health_risk_bounds = [(-1, 1)] * len(health_risk_feat)

# Makes a matrix including only featured values 
air_data = df[air_qual_feat].values
health_risk_data = df[health_risk_feat].values

# Scale featured values
scaler_air = StandardScaler()
scaler_health = StandardScaler()

air_data_scaled = scaler_air.fit_transform(air_data)
health_risk_data_scaled = scaler_health.fit_transform(health_risk_data)

# Creates a vetor of target values 
air_target_vals = df["airQuality"].values
health_risk_vals = df["healthRiskScore"].values

# Scale target values
scaler_air_target = StandardScaler()
scaler_health_target = StandardScaler()

air_target_scaled = scaler_air_target.fit_transform(air_target_vals.reshape(-1, 1)).flatten()
health_target_scaled = scaler_health_target.fit_transform(health_risk_vals.reshape(-1, 1)).flatten()

# Run simulated annealling
best_weight_air, best_eval_air = sim_ann(air_data_scaled, air_target_scaled, air_qual_bounds)
best_weight_health, best_eval_health = sim_ann(health_risk_data_scaled, health_target_scaled, health_risk_bounds)

# Run gradient descent
best_weight_air, best_eval_air = gradient_descent(air_data_scaled, air_target_scaled, np.array(best_weight_air))
best_weight_health, best_eval_health = gradient_descent(health_risk_data_scaled, health_target_scaled, np.array(best_weight_health))

# Implement random forest
rf_air = RandomForestRegressor(n_estimators=100, random_state=42)
rf_air.fit(air_data_scaled, air_target_scaled)
rf_pred_air = rf_air.predict(air_data_scaled).reshape(-1, 1)

rf_health = RandomForestRegressor(n_estimators=100, random_state=42)
rf_health.fit(health_risk_data_scaled, health_target_scaled)
rf_pred_health = rf_health.predict(health_risk_data_scaled).reshape(-1, 1)

# Combine linear + RF predictions as features
linear_pred_air = predict(best_weight_air, air_data_scaled).reshape(-1, 1)
linear_pred_health = predict(best_weight_health, health_risk_data_scaled).reshape(-1, 1)

stack_air = np.hstack([linear_pred_air, rf_pred_air])
stack_health = np.hstack([linear_pred_health, rf_pred_health])

# Meta-model
meta_air = LinearRegression()
meta_air.fit(stack_air, air_target_scaled)
stacked_air_pred = meta_air.predict(stack_air)

meta_health = LinearRegression()
meta_health.fit(stack_health, health_target_scaled)
stacked_health_pred = meta_health.predict(stack_health)

# Function to calculate r2
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Final Evaluation 
final_mse_air = mse(air_target_scaled, stacked_air_pred)
final_mse_health = mse(health_target_scaled, stacked_health_pred)

final_r2_air = r2_score(air_target_scaled, stacked_air_pred)
final_r2_health = r2_score(health_target_scaled, stacked_health_pred)

print(f"Air Quality - MSE: {final_mse_air:.5f}, R²: {final_r2_air:.5f}")
print(f"Health Risk - MSE: {final_mse_health:.5f}, R²: {final_r2_health:.5f}")
