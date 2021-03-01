
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the coefficient
of determination are also calculated.

"""
from sklearn import linear_model

print(__doc__)

# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
# from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

m = 500
np.random.seed(seed=5)
X = 6 * np.random.random(m).reshape(-1, 1) - 3
y = 0.5 * X ** 5 - X ** 3 - X ** 2 + 2 + 5 * np.random.randn(m, 1)

# Load the diabetes dataset
# X_old, y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
# X = X_old[:, np.newaxis, 2]
#
# Split the data into training/testing sets
X_train = X[:-300]
X_test = X[-200:]

# Split the targets into training/testing sets
y_train = y[:-300]
y_test = y[-200:]

# Create linear regression object
# regr = LinearRegression()
#
# # Train the model using the training sets
# regr.fit(X_train,
#          y_train)
#
# # Make predictions using the testing set
# y_pred = regr.predict(X_test)
#
# regr2 = LinearRegression()
# poly2_features = PolynomialFeatures(degree=10)
# X_poly2 = poly2_features.fit_transform(X_train)
# X_poly2_test = poly2_features.fit_transform(X_test)
# regr2.fit(X_poly2,
#           y_train)
# y_pred2 = regr2.predict(X_poly2_test)
#
# # The coefficients
# # print('Coefficients: \n', regr.coef_)
# # print('Itercept: \n', regr.intercept_)
# # The mean squared error
# print('Mean squared error of linear model: %.2f'
#       % mean_squared_error(y_test, y_pred))
# print('Mean squared error of poly2 model: %.2f'
#       % mean_squared_error(y_test, y_pred2))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
#       % r2_score(y_test, y_pred))
# print('Coefficient of determination ploy2: %.2f'
#       % r2_score(y_test, y_pred2))
#
# # Plot outputs
# plt.scatter(X_test, y_test, color='black')
# plt.plot(X_test, y_pred, color='blue', linewidth=3)
# plt.plot(X_test, y_pred2, color='red', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()

################################

loss_allmodels = []
loss_allmodels_train = []
mydegree = 25

for i in range(1, mydegree, 1):

    regr = linear_model.LinearRegression().fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    poly_features = PolynomialFeatures(degree=i)
    X_poly_train = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.fit_transform(X_test)

    model2 = LinearRegression()
    model2.fit(X_poly_train, y_train)
    y_pred2 = model2.predict(X_poly_test)
    y_pred2_train = model2.predict(X_poly_train)
    # print(model2.coef_)
    error_train = (y_pred2_train - y_train) ** 2
    error = (y_pred2 - y_test) ** 2
    loss = np.sum(error) / len(error)
    loss_train = np.sum(error_train) / len(error_train)
    loss_allmodels.append(loss)
    loss_allmodels_train.append(loss_train)

    error_train = (y_train - y_train) ** 2
    error = (y_pred2 - y_test) ** 2
    loss = np.sum(error) / len(error)
    loss_train = np.sum(error_train) / len(error_train)
    loss_allmodels.append(loss)
    loss_allmodels_train.append(loss_train)

    plt.figure()
    plt.scatter(X_test, y_test, color='red')
    plt.scatter(X_test, y_pred, color='blue')
    plt.scatter(X_test, y_pred2, color='green')
    plt.title('green-poly, blue-linear. degree:%d' % (i) +
              'test loss:%d' % loss +
              'train loss:%d' % loss_train)
    plt.show()

plt.figure()
plt.plot(range(len(loss_allmodels)), loss_allmodels, color='red')
plt.plot(range(len(loss_allmodels_train)), loss_allmodels_train, color='green')
plt.title('green-train, red-test')
plt.show()
