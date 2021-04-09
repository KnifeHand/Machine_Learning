#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""
Create on Fri March 4 12:17 2021
@author: Matt Hurt
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age',
             'labelvalue']

# load dataset:
# 1 select the pim diabetes dataset w/ binary target values
# 2. Use pandas to read CSV file as dataframe.
pima = pd.read_csv("pima-indians-diabetes.csv",
                   header=None, names=col_names)
print(pima.head(), '\n')

# split dataset in features and target variable
feature_cols = ['pregnant', 'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols]  # pregnant
# X3 = pima[0:3]
# X4 = pima.loc[0:3, ['pregnant', 'insulin']]  # 0:3 -> 4 rows
# X5 = pima.iloc[0:3, 0:1]  # 0:3 -> 3 rows, one column
y = pima.labelvalue  # Target variable


# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# import the class
# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('data model:\n', y_pred, '\n')

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cnf_matrix)

""" Import required modules"""
class_names = [0, 1]  # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

"""Create heatmap"""
sns.set(font_scale=1.0)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
# sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred), '\n')
print(metrics.classification_report(y_test, y_pred))

# ax = plt.figure()
# # ax.xticks('red')
# y_pred_proba = logreg.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
# auc = metrics.roc_auc_score(y_test, y_pred_proba)
# plt.plot(fpr, tpr, label="data 1, auc="+str(auc), color='green')
# plt.plot([0, 1], [0, 1], 'k--', color='red')
# plt.axis([0, 1, 0, 1])
# plt.title('Logistic Regression for Binary Classification')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True positive Rate')
# plt.legend(loc=4)
# plt.show()
plt.figure()
ax = fig.add_subplot(111, label="1")
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.figure().set_facecolor('black')
ax = plt.axes()
ax.set_facecolor('black')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
plt.plot(fpr, tpr, label="data 1, auc="+str(auc), color='cyan')
plt.plot([0, 1], [0, 1], 'k--', color='green')
plt.axis([0, 1, 0, 1])
title = plt.title('Logistic Regression for Binary Classification', )
title.set_color('white')
xlabel = plt.xlabel('False Positive Rate')
ylabel = plt.ylabel('True positive Rate')
xlabel.set_color("white")
ylabel.set_color("white")
plt.legend(loc=4)
plt.show()
