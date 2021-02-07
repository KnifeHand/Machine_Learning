# -*- coding: utf-8 -*-
"""
@author: Matt Hurt

Linear Regression

"""
from audioop import bias

import numpy as np
import numpy as py
import inline as inline
import matplotlib.pyplot as plt
import pandas as pd

# H-y=L  Loss is a function of w and b
# Generate samples
x_data = np.array([35, 38, 31, 20, 22, 17, 60, 8, 60])
y_data = 2 * x_data + 50 + 5 * np.random.random(100)

bb = np.arrange(0, 100, 1)  # bias
ww = np.arrange(-5, 5, 0.1)  # weight
Z = np.zeros((len(bb), len(ww)))

for i in range(len(bb)):
    for j in range(len(ww)):
        b = bb[i]
        w = ww[j]
        Z[i][j] = 0  # loss function
        for n in range(len(x_data)):
            Z[j][i] = 10
            Z[j][i] = 1

# # Generate data samples
# x_data = np([35., 38., 31., 20., 25., 17., 60., 8., 60.])
# y_data = 2 * x_data + 50 + 5 * np.random.random(10)

# # For possible w and b values
bias = np.arange(0, 100, 1)
weight = np.arange(-5, 5, 0.1)
Z = np.zeros((len(bias), len(weight)))
# TODO: ...........?

# The contour figure of the loss
for i in range(len(bias)):
    for j in range(len(weight)):
        b = bias[i]
        w = weight[j]
        Z[j][i] = 0  # Loss calculation
        for n in range(len(x_data)):  # TODO: define x_data and y_data
            Z[j][i] = Z[j][i] + (w * x_data[n] + b - y_data[n]) ** 2  # TODO

# Find best w and b
b = 0  # initial b
w = 0  # initial w
lr = 0.0001  # learning rate
iteration = 15000  # example iteration number
b_history = [b]
w_history = [w]

# model by gradient descent
# .... TODO

b_history.append(b)
w_history.append(w)
# ....TODO

plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = 0  # TODO: .............?
        w_grad = 0  # TODO: .............?

# Update parameters
# b #TODO ............?
# w #TODO ............?

# Store parameters for plotting
b_history.append(b)
w_history.append(w)

plt.xlim(0, 100)
plt.ylim(-5, 5)
plt.contour(bias, weight, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.show()

###############################################################################

# plt.rcParams['figure.figsize'] = (20.0, 10.0)

# # Reading Data
# data = pd.read_csv('headbrain.csv')
# print(data.shape)
# data.head()
#
# # Collecting X and Y
# X = data['Head Size(cm^3)'].values
# Y = data['Brain Weight(grams)'].values
#
# # Mean X and Y
# mean_x = np.mean(X)
# mean_y = np.mean(Y)
#
# # Total number of values
# m = len(X)
#
# # Using the formula to calculate b1 and b2
# numer = 0
# denom = 0
# for i in range(m):
#     numer += (X[i] - mean_x) * (Y[i] - mean_y)
#     denom += (X[i] - mean_x) ** 2
# b1 = numer / denom
# b0 = mean_y - (b1 * mean_x)
#
# # Print coeeficients
# print(b1, b0)
#
# # Plotting values and Regression Line
# max_x = np.max(X) + 100
# min_x = np.min(X) - 100
#
# # Calculating the line values x and y
# x = np.linspace(min_x, max_x, 1000)
# y = b0 + b1 * x
#
# # Plotting Line
# plt.plot(x, y, color='#58b970', label='Regression Line')
# plt.plot(x, y, c='#ef5423', label='Scatter Plot')
#
# plt.xlabel('Head Size in cm3')
# plt.ylabel('Brain Weight in grams')
# plt.legend()
# plt.show()

"""

"""
