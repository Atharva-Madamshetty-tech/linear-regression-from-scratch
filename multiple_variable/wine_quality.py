import numpy as np
import pandas as pd

# Load the data

df = pd.read_csv('winequality-red.csv')

# Input the training data

x_train = df.drop('quality',axis = 1).values
y_train = df['quality'].values

# Cost Function

def compute_cost(X,y,w,b):
  m = X.shape[0]
  cost = 0
  for i in range(m):
    f_wb = np.dot(X[i],w) + b
    cost_t = (f_wb - y[i]) ** 2
    cost = cost + cost_t
  total_cost = (1/(2*m)) * cost
  return total_cost


# COmputation of Gradient


def compute_gradient(X,y,w,b):
  m = X.shape[0]
  n = X.shape[1]
  dj_dw = np.zeros((n,))
  dj_db = 0
  for i in range(m):
    f_wb = np.dot(X[i],w) + b
    err = f_wb - y[i]
    for j in range(n):
      dj_dw[j] = dj_dw[j] + err * X[i,j]
    dj_db = dj_db + err
  dj_dw = dj_dw/m
  dj_db = dj_db/m
  return dj_dw,dj_db

# Gradient Descent

def gradient_descent(X,y,w_in,b_in,alpha,num_iter,compute_cost,compute_gradient):
  m = X.shape[0]
  w = w_in
  b = b_in
  for i in range(num_iter):
    dj_dw,dj_db = compute_gradient(X,y,w,b)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db          
    if i%(num_iter // 10) == 0:
      cost = compute_cost(X,y,w,b)
      print(f"Iterations: {i:4}, Cost: {cost:.6f}, w: {w}, b: {b:.3f}")
  return w,b


# Input the values for prediction


w_final,b_final = gradient_descent(x_train,y_train,np.zeros(x_train.shape[1]),0,0.00001,10000,compute_cost,compute_gradient)

# Enter the features


print("Enter the features")
fa = float(input("Enter the fixed acidity"))
va = float(input("Enter the volatile acidity in the wine"))
cs = float(input("Enter the citric acid in the wine "))
rsu = float(input("Enter the residual sugar in the wine "))
ch = float(input("Enter the chlorides in the wine"))
fsd = int(input("Enter the free sulfur dioxide"))
tsd = int(input("Enter the total sulfur dioxide"))
de = float(input("Enter the density of the wine"))
ph = float(input("Enter the pH of the wine"))
sh = float(input("Enter the sulphate content in the wine"))
al = float(input("Enter the alcohol content in the wine"))

x_input = np.array([fa,va,cs,rsu,ch,fsd,tsd,de,ph,sh,al])
prediction = (np.dot(x_input,w_final)) + b_final
print(f"The Quality of the wine according to the features specified is: {prediction}")

