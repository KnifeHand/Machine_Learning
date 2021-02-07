# -*- coding: utf-8 -*-
"""
@author: Matt Hurt

Linear Regression

"""
import numpy as np
import matplotlib.pyplot as plt

# Generate samples
x_data = np.array([35.,38.,31.,20.,22.,17.,60.,8.,60.])
y_data = 2*x_data+50+5*np.random.random()

# Plot the landscape
bb = np.arange(0,100,1)# bias
ww = np.arange(-5,5,0.1)# weight
Z = np.zeros((len(bb),len(ww)))

# # # For possible w and b values
# bias = np.arange(0, 100, 1)
# weight = np.arange(-5, 5, 0.1)
# Z = np.zeros((len(bias), len(weight)))

for i in range(len(bb)):
    for j in range(len(ww)):
        b = bb[i]
        w = ww[j]
        Z[j][i] = 0  # loss function
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (w*x_data[n]+b - y_data[n])**2 # this is the loss
            Z[j][i] = Z[j][i]/len(x_data)

# Find best w and b
b = 0  # initial b
w = 0  # initial w
lr = 0.001  # learning rate
iteration = 10000  # example iteration number

# Iterations
b_history = [b]
w_history = [w]

# The contour figure of the loss
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad += b_grad+(b+w*x_data[n]-y_data[n])*1.0
        w_grad += w_grad+(b+w*x_data[n]-y_data[n])*x_data[n]
    b = b-b_grad*lr
    w = w-w_grad*lr

# model by gradient descent

# Store parameters for plotting
b_history.append(b)
w_history.append(w)

plt.xlim(0,100)
plt.ylim(-5,5)
plt.contour(bb, ww, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.contour(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
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


