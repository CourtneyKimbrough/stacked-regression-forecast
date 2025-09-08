import numpy as np
import random
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

def sim_ann(data, target_vals, bounds, n_iterations=1000, step_size=0.1, temp=10):
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

best_weight_air, best_eval_air = sim_ann(air_data_scaled, air_target_scaled, air_qual_bounds)
best_weight_health, best_eval_health = sim_ann(health_risk_data_scaled, health_target_scaled, health_risk_bounds)


print("Air Quality Feature Weights:")
for feature, weight in zip(air_qual_feat, best_weight_air):
    print(f"{feature}: {weight:.3f}")
print(f"Evaluation: \033[92m{best_eval_air:.5f}\033[0m\n")

print("Health Risk Feature Weights:")
for feature, weight in zip(health_risk_feat, best_weight_health):
    print(f"{feature}: {weight:.3f}")
print(f"Evaluation: \033[92m{best_eval_health:.5f}\033[0m")