import numpy as np
import pandas as pd


# Load the dataset from a CSV file

df = pd.read_csv('YOur_dataset.csv')

x_train = df['Study Hours'].values  # Feature: Study Hours
y_train = df['Exam Score (out of 100)'].values  # Target: Exam Score

# Cost Function

def compute_cost(x, y, w, b):
   
    m = x.shape[0]  # Number of training examples
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b  # Model's prediction
        cost_t = (f_wb - y[i]) ** 2  # Squared error for a single example
        cost += cost_t
    total_cost = (1 / (2 * m)) * cost  # Compute the total cost (Mean Squared Error)

    return total_cost

# Gradient Computation
def compute_gradient(x, y, w, b):
   
    m = x.shape[0]
    dj_dw = 0  # Gradient for w
    dj_db = 0  # Gradient for b

    for i in range(m):
        f_wb = w * x[i] + b  # Model's prediction
        dj_dw_t = (f_wb - y[i]) * x[i]
        dj_db_t = (f_wb - y[i])
        dj_dw += dj_dw_t
        dj_db += dj_db_t
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


# Calculating gradient Descent


def gradient_descent(x, y, w_in, b_in, alpha, num_iter, compute_cost, compute_gradient):
   
    w = w_in
    b = b_in
    for i in range(num_iter):
       
      # Calculate the gradient and update the parameters
      
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Print cost every 10% of the iterations for monitoring
      
        if i % (num_iter // 10) == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iterations {i:4}, Cost : {cost:.6f}, w : {w:.3f}, b : {b:.3f}")

    return w, b

# Run the gradient descent algorithm to find the optimal w and b

w_final, b_final = gradient_descent(x_train, y_train, 0, 0, 0.001, 100000, compute_cost, compute_gradient)

# Make a prediction for a new input
try:
    a = float(input("Enter the hours that the student has studied: "))
    prediction = (w_final * a) + b_final
    print(f"If the student studied {a} hours, they will probably get {prediction:.1f} number of marks.")
except ValueError:
    print("Invalid input. Please enter a number.")



