# -*- coding: utf-8 -*-
"""
@author: Matt Hurt

Linear Regression via Adarsh Menon

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7.0, 5.0)

# Preprocess the input data
# data = pd.read_csv('random_number_gen_2cols.csv')
rand = np.random.randint(2, 127, size=(100, 2))
df = pd.DataFrame(rand, columns=['random_numbers_1', 'random_numbers_2']).astype(float)

data = df

X = data.iloc[:, 0]
Y = data.iloc[:, 1]
# plt.scatter(X,Y)
# plt.show()

# Build the model
m = 0
c = 0

L = 0.0001
epochs = 1000

n = float(len(X))

for i in range(len(X)):
    Y_pred = m*X + c
    D_m = (-2/n) * sum(X * (Y - Y_pred))
    D_c = (-2/n) * sum(Y - Y_pred)
    m = m - L*D_m
    c = c - L*D_c

print(m,c)

# Make predictions
Y_pred = m*X + c

plt.scatter(X,Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
# plt.scatter(X, Y_pred)
plt.show()

###############################################################################

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
