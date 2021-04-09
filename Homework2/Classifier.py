# #!/usr/bin/env python3
# # _*_ coding: utf-8 _*_
# """
# Create on Fri March 4 12:17 2021
# @author: Matt Hurt
# """
# import pandas as pd
# col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age',
#              'labelvalue']
#
# # load dataset:
# # 1 select the pim diabetes dataset w/ binary target values
# # 2. Use pandas to read CSV file as dataframe.
# pima = pd.read_csv("pima-indians-diabetes.csv",
#                    header=None, names=col_names)
# print(pima.head(), '\n')
#
# # split dataset in features and target variable
# feature_cols = ['pregnant', 'age', 'glucose', 'bp', 'pedigree']
# X = pima[feature_cols]  # pregnant
# y = pima.labelvalue  # Target variable
#
# """Create a test set """
# X_train, X_test, y_train, y_test = X[0:200], X[0:1], y[0:200], y[0:1]
#
# """Shuffle the training set.  This will gurantee that all cross-validation folds
# will be similar"""
# import numpy as np
# shuffle_index = np.random.permutation(100)
# X_train,y_train = X_train[shuffle_index], y_train[shuffle_index]
#
# """Training a Binary Classifier.  Simplify the proble for now and only try to
# identify one feature.  For example - pregnant or not-pregnant.  Create the
# target vectors for this classification"""
# y_train_pregnant = feature_cols
# y_test_pregnant = y_test
#
# """Now pick a classifier and train it.  A good place to start is with the
# Stochastic Gradient Descent (SGD) classifier.  This will handle very large
# datasets because it deals with dataset independently, one at a time."""
# from sklearn.linear_model import SGDClassifier
# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_pregnant)
#
