import numpy as np
import pandas as pd

# Load the dataset

df = pd.read_csv('stock_price_dataset.csv')

# Apply log transformation to 'Volume' to scale large values

df['Volume'] = np.log(df['Volume'])

# Prepare feature matrix (X) and target vector (y)

x_train = df.drop('Close', axis=1).values  # Features: all columns except 'Close'
y_train = df['Close'].values  # Target: 'Close' column

# Function to compute Mean Squared Error cost

def compute_cost(X, y, w, b):
    m = X.shape[0]  # Number of training examples
    cost = 0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b    # Predicted value
        cost_t = (f_wb - y[i]) ** 2   # Squared error
        cost += cost_t  # Accumulate cost
    total_cost = (1 / (2 * m)) * cost  # Mean squared error divided by 2
    return total_cost

# Function to compute gradients with respect to weights and bias

def compute_gradient(X, y, w, b):
    m = X.shape[0]    # Number of examples
    n = X.shape[1]    # Number of features
    dj_dw = np.zeros((n,))    # Gradient for weights
    dj_db = 0    # Gradient for bias
    for i in range(m):
        f_wb = np.dot(X[i], w) + b    # Predicted value
        err = f_wb - y[i]    # Error
        for j in range(n):
            dj_dw[j] += err * X[i, j]    # Accumulate gradient for each weight
        dj_db += err    # Accumulate gradient for bias
    dj_dw /= m    # Average over all examples
    dj_db /= m
    return dj_dw, dj_db

# Function to perform gradient descent optimization

def gradient_descent(X, y, w_in, b_in, alpha, num_iter, compute_cost, compute_gradient):
    w = w_in  # Initialize weights
    b = b_in  # Initialize bias
    for i in range(num_iter):
        dj_dw, dj_db = compute_gradient(X, y, w, b)  # Compute gradients
        w -= alpha * dj_dw  # Update weights
        b -= alpha * dj_db  # Update bias
        # Print cost every 10% of total iterations
        if i % (num_iter // 10) == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Iterations: {i:4}, Cost: {cost:.6f}, w: {w}, b: {b}")
    return w, b

# Run gradient descent to find optimized weights and bias

w_final, b_final = gradient_descent(
    x_train, y_train, 
    np.zeros(x_train.shape[1]),  # Initialize weights as zeros
    0,  # Initialize bias as zero
    0.00001,  # Learning rate
    10000,  # Number of iterations
    compute_cost, compute_gradient
)

# Take user input for predicting stock closing price

print("Enter the features")
open_ = float(input("Enter the opening rate of the stock: "))
high_ = float(input("Enter the highest price of the stock: "))
low_ = float(input("Enter the lowest value of the stock: "))
volume = float(input("Enter the volume of the stock: "))
mov5 = float(input("Enter the Moving Avg 5 of the stock: "))
mov10 = float(input("Enter the Moving Avg 10 of the stock: "))
vol = float(input("Enter the volatility of the stock: "))

# Apply log transformation to input volume (to match training data scaling)

v = np.log(volume)

# Prepare input array for prediction

x_input = np.array([open_, high_, low_, v, mov5, mov10, vol])

# Predict the closing price using trained weights and bias
prediction = np.dot(x_input, w_final) + b_final

print(f"The closing price of the stock may be: {prediction} INR")
