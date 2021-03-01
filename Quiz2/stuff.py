# -*- coding: utf-8 -*-
def cls(): return print("\033[2J\033[;H", end='')


cls()

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

max_var = 500
np.random.seed(seed=5)
X = 6 * np.random.random(max_var).reshape(-1, 1) - 3
y = 0.5 * X ** 5 - X ** 3 - X ** 2 + 2 + 5 * np.random.randn(max_var, 1)

X_train = X[:-300]
X_test = X[-200:]
Y_train = y[:-300]
Y_test = y[-200:]

# MSEL = Mean squared error loss
total_loss = []
training_loss = []

for i in range(1, 25, 1):  # 2- 25 degrees inclusive

    regr = linear_model.LinearRegression().fit(X_train, Y_train)
    y_prediction = regr.predict(X_test)

    poly_features = PolynomialFeatures(degree=i)
    X_poly = poly_features.fit_transform(X_train)
    X_poly_test = poly_features.fit_transform(X_test)
    regr2 = LinearRegression().fit(X_poly, Y_train)
    y_prediction2 = regr2.predict(X_poly_test)
    y_prediction2_train = regr2.predict(X_poly)

    MSEL_train = (y_prediction2_train - Y_train) ** 2
    MSEL_test = (y_prediction2 - Y_test) ** 2
    loss_test = np.sum(MSEL_test) / len(MSEL_test)
    loss_train = np.sum(MSEL_train) / len(MSEL_train)
    total_loss.append(loss_test)
    training_loss.append(loss_train)

    if i == 2 or i == 5 or i == 8 or i == 10 or i == 20 or i == 3 or i == 4 or i == 1:
        plt.scatter(X_test, Y_test, color='red', label='scatter plot')
        plt.plot(X_test, y_prediction, color='blue', lw=3, label='linear')
        plt.plot(X_test, y_prediction2, 'o-', ms=2, lw=0, color='green', label='polynomial')
        plt.legend(loc='lower right')

        plt.xticks(())
        plt.yticks(())
        plt.title('Current degree:%d' % (i) + '-Testing loss:%d' % (loss_test) + '-Training loss:%d' % (loss_train))
        plt.show()

plt.figure()
plt.plot(range(len(total_loss)), total_loss, color='red', label='testing loss')
plt.plot(range(len(training_loss)), training_loss, color='blue', label='training loss')
plt.show()

plt.legend(loc='upper right')
plt.title('Training vs Testing Loss from 2-25 degrees')

print('Coefficients: \n', regr.coef_)
print('Itercept: \n', regr.intercept_)
print('Mean squared error: %.2f' % mean_squared_error(Y_test, y_prediction))
print('Coefficient of determination: %.2f' % r2_score(Y_test, y_prediction))
