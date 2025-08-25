"""
House Price Prediction using Linear Regression (Single Variable)

This script implements linear regression from scratch using only NumPy and Pandas.
It predicts house prices based on the area of the house.

Key steps:
1. Load dataset (area vs price)
2. Define cost function (Mean Squared Error)
3. Compute gradients
4. Run gradient descent to optimize parameters
5. Predict house prices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load Dataset
# Dataset must contain two columns: "area" and "price"

df = pd.read_csv('house-prices.csv')


# Cost Function

def compute_cost(x, y, w, b):
  
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b   # Prediction
        cost_t = (f_wb - y[i]) ** 2
        cost += cost_t
    total_cost = (1 / (2 * m)) * cost
    return total_cost


# Gradient Computation

def compute_gradient(x, y, w, b):
   
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b   # Prediction
        dj_dw_t = (f_wb - y[i]) * x[i]
        dj_db_t = (f_wb - y[i])
        dj_dw += dj_dw_t
        dj_db += dj_db_t
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db    


# Gradient Descent

def gradient_descent(x, y, w_in, b_in, alpha, num_iter, compute_cost, compute_gradient):
   
    w = w_in
    b = b_in

    for i in range(num_iter):
        # Compute gradient
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Update parameters
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Print progress every 10% of iterations
        if i % (num_iter // 10) == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {i:4}: Cost = {cost:.6f}, w = {w:.3f}, b = {b:.3f}")
       
    return w, b    


# Training Data

x_train = df['area'].values   # Feature: house area (sqft)
y_train = df['price'].values  # Target: house price


# Train the Model

w_final, b_final = gradient_descent(
    x_train, y_train,
    w_in=0, b_in=0,
    alpha=0.0000001, num_iter=1000,
    compute_cost=compute_cost,
    compute_gradient=compute_gradient)


# Make Predictions

a = int(input("Enter the size of the house in square feet"))
predicted_price = (w_final * a) + b_final
print(f"Predicted price of a {a} sqft house: ${predicted_price:.1f}")
