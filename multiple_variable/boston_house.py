"""
Boston Housing Price Prediction using Linear Regression (Multiple Variables)

This script implements multivariable linear regression from scratch 
to predict the median house price (medv) based on multiple features. 
It uses gradient descent to optimize weights (w) and bias (b).
"""


import numpy as np
import pandas as pd

# Load Dataset
df = pd.read_csv('BostonHousing.csv')
df = df.drop('b' , axis = 1)          # Drop unnecessary column 'b' (not used in training)

# Separate features (X) and target (y)

x_train = df.drop('medv', axis = 1).values    # Feature matrix
y_train = df['medv'].values                   # Target vector (house price)
 

# Cost Function

def compute_cost(X,y,w,b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb = np.dot(X[i] , w) + b
        cost_t = (f_wb - y[i]) ** 2
        cost = cost + cost_t
    total_cost = (1/(2*m)) * cost    # Squared error
    return total_cost

# Gradient Computation

def compute_gradient(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n,))     # Gradient for each feature weight
    dj_db = 0
    for i in range(m):
        f_wb = np.dot(X[i] , w) + b
        err = (f_wb - y[i])
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i,j]
        dj_db = dj_db + err
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw,dj_db

# Gradient Descent

def gradient_descent(X,y,w_in,b_in,alpha,num_iter,compute_cost,compute_gradient):
    w = w_in
    b = b_in
    m = X.shape[0]
    for i in range(num_iter):
      # Compute gradient
        dj_dw,dj_db = compute_gradient(X,y,w,b)
        w = w - alpha * (dj_dw)
        b = b - alpha * (dj_db)
       # Print cost at intervals
  
      if i%(num_iter // 10) == 0:
            cost = compute_cost(X,y,w,b)
            print(f"Iterations{i:4}: Cost:{cost:.6f}, w:{w}, b:{b:.3f}")
    return w,b    

# Train Model

w_final,b_final = gradient_descent(x_train,y_train,np.zeros(x_train.shape[1]),0,0.000001,10000,compute_cost,compute_gradient)

# User Input for Prediction

print("Enter the features for Prediction")
crim = float(input("Enter the per capita crime rate by town"))
zn = float(input("Enter the proportion of residential land zoned for lots over 25,000 sq.ft."))
indus = float(input("Enter the proportion of non-retail business acres per town."))
chas = int(input("Enter the Charles River dummy variable (1 if tract bounds river, 0 otherwise)"))
nox = float(input("Enter the  nitric oxides concentration (parts per 10 million)"))
rm = float(input("Enter the average number of rooms per dwelling"))
age = float(input("Enter the proportion of owner-occupied units built prior to 1940"))
dis = float(input("Enter the weighted distances to five Boston employment centres"))
rad = int(input("Enter the index of accessibility to radial highways"))
tax = int(input("Enter the full-value property-tax rate per $10,000"))
ptratio = float(input("Enter the pupil-teacher ratio by town"))
lstat = float(input("Enter the % lower status of the population"))

inp = np.array([crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,lstat])
print(f"The {inp} Sqft house : ${np.dot(w_final, inp) + b_final:.1f}") 

