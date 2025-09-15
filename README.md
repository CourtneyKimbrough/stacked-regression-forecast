# Stacked Regression Forecast

This project demonstrates an AI solution that predicts **air quality** and **health risk scores** using a stacked regression approach. The model combines **gradient descent-based linear regression** with a **Random Forest Regressor** in a meta-model, allowing improved predictive performance and interpretability.

## Features

- **Gradient Descent Linear Regression:** Optimized with L1 (Lasso) and L2 (Ridge) regularization.
- **Random Forest Regressor:** Captures non-linear relationships between features and targets.
- **Stacked Meta-Model:** Combines predictions from linear and ensemble models to enhance accuracy.
- **Feature Analysis:** Outputs feature weights and importance to provide insights for informed decision-making.
- **Evaluation Metrics:** MSE (Mean Squared Error) and R² to assess model performance.

## Dataset

The model uses a dataset containing environmental factors (e.g., `pm2.5`, `no2`, `co2`, temperature, humidity, wind gust) and target variables (`airQuality`, `healthRiskScore`).  

