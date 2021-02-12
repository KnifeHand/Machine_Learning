# -*- coding: utf-8 -*-
"""
@author: Matt Hurt

Reference: "Hands-On Machine Learning with Scikit-Learn & TensorFlow"
Author: Aurélien Géron
Publisher: Oreilly 2017

**Chapter 4: Training Models**
***Linear Regression using Gradient Decent***
To find the value of theta that minimizes the cost function, there is a
closed-form solution which is an equation that gives the result directly
called the Normal Equation.

Normal Equation: ThetaPrime = (X_Transpose * X) **-1 * X_Transpose * y

* ThetaPrime is the value of Theta that minimizes the cost funciton.
* y is the vector of target values containing y**(1) to y**(m).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = (7.0, 5.0)

##############################Normal Equation##################################
# Generate some linear-looking data to test this equation
#   ThetaPrime = (X_Transpose * X) **-1 * X_Transpose * y
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

""" 
Now compute ThetaPrime using the Normal Equation.  Here the inv() function 
from NymPy's Linear Algebra module (np.linalg) to compute the inverse of a 
matrix, and the dot() method for matrix multiplication:
"""
X_b = np.c_[np.ones((100, 1)), X]  # add x_sub0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# The actual function that is used to generate the data is y = 4 + 3*x_sub0 + Gaussian Noise.
# This print statement will show what the equation found.
print("\nTheta Best: ")
print(theta_best), print("\n")

"""
Notice for Theta_sub0 = 4.17478291.  It would have been nice to have the 
calculation come in at 4 but this is close enough.  However, the noise made it 
impossible to recover the exact parameters of the original function.
"""

# Now, predictions can be made using ThetaPrime.
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # again, add x_sub0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)

print("New Prediction: ")
print(y_predict), print("\n")

# Plotting this model's predictions
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

"""
Batch Gradient Descent

To implement Gradient Descent, we need to calculate how much the cost function 
will change if you change Theta_sub_j just a little bit.  The official definition
to implement Gradient Descent is computing the gradient of the cost function with
regards to each model parameter Theta_sub_j.  This is called the partial derivative.
In other words, "what is the slope of the mountain under my feet if I face east?" and
then asking the same question when facing any other direction. 

Instead of computing the gradients individually, the Gradient vector of the cost 
function will help do this all at once.  The calculations are over the full 
training set X, at each Gradient Descent step!  This algorithm uses the whole
batch of training data at every step however, getting the results takes time 
because the training sets are very large.  

Once the gradient vector is discovered, which points uphill, just go in the 
opposite direction to go downhill.  This means subtracting the change of Partial 
derivatives of the cost function from Theta.  This is where the learning rate
comes into play.  

This can be achieved by multiplying the gradient vector by the learning rate to
determine the size of the downhill step
"""

###############################Gradient Descent step###########################
eta = 0.1  # learning rate
n_iterations = 10000
m = 100

theta = np.random.randn(2, 1)  # random initialization

for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print("Gradient Descent: "), print(theta), print('\n')

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

#############################Stochastic Gradient Descent#######################
"""
The main problem with Batch Gradient Descent is the fact that it uses the whole
training set to compute the gradients at every step, which makes it very slow when
the training set is large. At the opposite extreme, Stochastic Gradient Descent just
picks a random instance in the training set at every step and computes the gradients
based only on that single instance. Obviously this makes the algorithm much faster
since it has very little data to manipulate at every iteration. It also makes it possible to
train on huge training sets, since only one instance needs to be in memory at each
iteration (SGD can be implemented as an out-of-core algorithm.7)

On the other hand, due to its stochastic (i.e., random) nature, this algorithm is much
less regular than Batch Gradient Descent: instead of gently decreasing until it reaches
the minimum, the cost function will bounce up and down, decreasing only on aver‐
age. Over time it will end up very close to the minimum, but once it gets there it will
continue to bounce around, never settling down (see Figure 4-9). So once the algo‐
rithm stops, the final parameter values are good, but not optimal.

When the cost function is very irregular (as in Figure 4-6), this can actually help the
algorithm jump out of local minima, so Stochastic Gradient Descent has a better
chance of finding the global minimum than Batch Gradient Descent does.

Therefore randomness is good to escape from local optima, but bad because it means
that the algorithm can never settle at the minimum. One solution to this dilemma is
to gradually reduce the learning rate. The steps start out large (which helps make
quick progress and escape local minima), then get smaller and smaller, allowing the
algorithm to settle at the global minimum. This process is called simulated annealing,
because it resembles the process of annealing in metallurgy where molten metal is
slowly cooled down. The function that determines the learning rate at each iteration
is called the learning schedule. If the learning rate is reduced too quickly, you may get
stuck in a local minimum, or even end up frozen halfway to the minimum. If the
learning rate is reduced too slowly, you may jump around the minimum for a long
time and end up with a suboptimal solution if you halt training too early.
This code implements Stochastic Gradient Descent using a simple learning schedule:
"""
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
    xi = X_b[random_index:random_index + 1]
    yi = y[random_index:random_index + 1]
    gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
    eta = learning_schedule(epoch * m + i)
    theta = theta - eta * gradients

print(theta)
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()

###########################Attempt 1###########################################
# data = pd.read_csv('random_number_gen_2cols.csv')
# rand = np.random.randint(2, 127, size=(100, 2))
# df = pd.DataFrame(rand, columns=['random_numbers_1', 'random_numbers_2']).astype(float)

# data = df
#
# X = data.iloc[:, 0]
# Y = data.iloc[:, 1]
# # plt.scatter(X,Y)
# # plt.show()
#
# # Build the model
# m = 0
# c = 0
#
# L = 0.0001
# epochs = 1000
#
# n = float(len(X))
#
# for i in range(len(X)):
#     Y_pred = m*X + c
#     D_m = (-2/n) * sum(X * (Y - Y_pred))
#     D_c = (-2/n) * sum(Y - Y_pred)
#     m = m - L*D_m
#     c = c - L*D_c
#
# print(m,c)
#
# # Make predictions
# Y_pred = m*X + c
#
# plt.scatter(X,Y)
# plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')
# # plt.scatter(X, Y_pred)
# plt.show()

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
