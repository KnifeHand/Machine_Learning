# -*- coding: utf-8 -*-
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
# import theta as theta
import pytest as pt
from numpy import number, ndarray
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.semi_supervised.tests.test_self_training import X_train, y_train
from sklearn.model_selection import learning_curve

m = 500
np.random.seed(seed=5)
X = 6 * np.random.random(m).reshape(-1, 1) - 3
y = 0.5 * X ** 5 - X ** 3 - X ** 2 + 2 + 5 * np.random.randn(m, 1)

model = LinearRegression()

###############################################################################
# create quadratic features
poly_features_quadratic = PolynomialFeatures(degree=2)
poly_features_cubic = PolynomialFeatures(degree=3)
X_quad_transform = poly_features_quadratic.fit_transform(X)
X_cubic_transform = poly_features_cubic.fit_transform(X)

# fit features
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
model = model.fit(X, y)

y_lin_fit = model.predict(X_fit)
linear_r2 = r2_score(y, model.predict(X))
model = model.fit(X_quad_transform, y)

y_quad_fit = model.predict(poly_features_quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, model.predict(X_quad_transform))
model = model.fit(X_cubic_transform, y)

y_cubic_fit = model.predict(poly_features_cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, model.predict(X_cubic_transform))

# plot results
plt.scatter(X, y, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit,
         label='linear (d=1), $R^2=%.2f$' % linear_r2,
         color='blue',
         lw=2,
         linestyle=':')
plt.plot(X_fit, y_quad_fit,
         label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red',
         lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit,
         label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
         color='green',
         lw=2,
         linestyle='--')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='lower right')
plt.show()
######################## Calculating the loss #################################
# def h(X, theta):
#     return X @ theta
# def J(theta, X, y):
#     return np.mean(np.square(h(X, theta) - y))
# alpha = 0.01
# theta = theta - alpha * (1/m) * (X.T @ ((X @ theta) - y))
# losses = []
# for _ in range(500):
#     theta = theta - alpha * (1/m) * (X.T @ ((X @ theta) - y))
#     losses.append(J(theta, X, y))
#
# predictions = h(X, theta)
# plt.plot(X[:, 1], predictions, label='predictions')
# plt.plot(X[:, 1], y, 'rx', label='labels')
# plt.legend()
######################## Diagnosing bias and variance #########################


pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', random_state=1))
train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=10,
    n_jobs=1)
train_mean: Union[number, number, ndarray] = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes,
         train_mean,
         color='blue',
         marker='o',
         markersize=5,
         label='training accuracy')
plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15,
                 color='blue')
plt.plot(train_sizes,
         test_mean,
         color='green',
         linestyle='--',
         marker='s',
         markersize=5,
         label='validation accuracy')
plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15,
                 color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()

######################## Diagnosing bias and variance #########################
from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name='logisticregression__C',
    param_range=param_range,
    cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range,
         train_mean,
         color='blue',
         marker='o',
         markersize=5,
         label='training accuracy')
plt.fill_between(param_range,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15,
                 color='blue')
plt.plot(param_range,
         test_mean,
         color='green',
         linestyle='--',
         marker='s',
         markersize=5,
         label='validation accuracy')
plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15,
                 color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.03])
plt.show()
