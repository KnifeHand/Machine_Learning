#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:13:39 2018

"""

##exercise on matplotlib

# matplotlib package has the plotting functionality of MATLAB
import matplotlib.pyplot as plt


# plt.plot generates a line on the graph from a list of
# x-coordinates and a list of y-coordinates.
x = range(101)
y1 = [xi**2 for xi in x]
# Use labels to name lines in the graph.
plt.plot(x, y1, label="x^2")

# Call plt.plot multiple times to plot multiple lines.
y2 = [xi**(1.5) for xi in x]
# Labels can be latex code.
plt.plot(x, y2, label=r"$x^{1.5}$")

# plt.xlabel and plt.ylabel assign names to the axes.
plt.xlabel(r"$x$", fontsize=16)
# plt.legend adds a legend to the graph with the line labels.
plt.legend(loc='upper left')

# plt.show displays the graph.
plt.show()


#####More exercise on Numpy
# NumPy namespace convention: np
import numpy as np
 
# Create an all-zero matrix
#   NOTE: argument is a tuple '(3, 4)'
#     WRONG: np.zeros(3, 4)
#     CORRECT: np.zeros( (3, 4) )
A = np.zeros( (3, 4) )
#A = np.ones((1,2))

# numpy creates arrays by default
# CVXPY works best with matrices
# It's best to cast every array to a matrix manually
#A = np.asmatrix(A)
print (A)
print (A.shape) # size of A

# All-one matrix
B = np.ones( (3, 4) )
B = np.asmatrix(B)
print (B)

# Identity matrix
I = np.asmatrix(np.eye(5))
print (I)

# Stacking matrices horizontally
#   Use vstack to stack vertically
J = np.hstack( (I, I) )
print (J)

# Random matrix with standard Gaussian entries
#   NOTE: argument is NOT a tuple
Q = np.random.randn(4, 4)
Q = np.asmatrix(Q)

print (Q)
print (Q[:, 1]) # Second column (everything is 0-indexed)
print (Q[2, 3]) # (3, 4) entry (as a real number)

# Random column vector of length 4
v = np.asmatrix(np.random.randn(4, 1))

# v.T: v tranpose
z = v.T * Q * v

# The result is a 1-by-1 matrix
print (z)

# Extract the result as a real number
print (z[0, 0])


# Other useful methods
#   Construct a matrix
A = np.matrix("1 2; 3 4")
B = np.matrix("-1 3.2; 5 8")
#   Transpose a matrix
print (A.T)
#   Elementwise multiplication
print (np.multiply(A, B))
#   Sum of each column (as a row vector)
print (np.sum(A, axis=0))
#   Sum of each row (as a column vector)
print (np.sum(A, axis=1))

# Linear algebra routines
Q = A.T*A
v = np.matrix("1 2").T
(d, V) = np.linalg.eig(Q) # Eigendecomposition
print ("d = ", d)
print ("V = ", V)

print ("||v||_2 = ", np.linalg.norm(v)) # 2-norm of a vector

Qinv = np.linalg.inv(Q) # Matrix inverse
# Solves Qx = v (faster than Qinv*v)
x = np.linalg.solve(Q, v)
print ("Q^{-1}v = ", x)


