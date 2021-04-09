"""
Author: Matt Hurt
References:
    1. https://www.youtube.com/watch?v=09mb78oiPkA
    2. https://www.kaggle.com/uciml/iris
    3. https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
    4. https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
    5. https://www.analyticsvidhya.com/blog/2014/10/introduction-k-neighbours-algorithm-clustering/
    6. https://www.kaggle.com/skalskip/iris-data-visualization-and-knn-classification
"""

print(__doc__)

import numpy as np
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from pandas.plotting import andrews_curves
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Importing the dataset
dataset = pd.read_csv('Iris.csv')

""" 
We can get a quick idea of how many instances (rows) and how many attributes
(columns) the data contains with the shape property.
"""
dataset.shape
print(dataset)
print('\n------------------------------------------------------------------------\n')
# dataset.head(5)
print(dataset.head(5))
print('\n------------------------------------------------------------------------\n')
# dataset.describe()
print("Description of the dataset")
print(dataset.describe())
print('\n------------------------------------------------------------------------\n')
print("Information about the dataset")
print(dataset.info(), '\n')
print('\n------------------------------------------------------------------------\n')
# Let’s now take a look at the number of instances (rows) that belong to each class.
# We can view this as an absolute count.
print("Number of instances to each class")
print(dataset.groupby('Species').size())
print('\n------------------------------------------------------------------------\n')


# Divide data into features and labels.
feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = dataset[feature_columns].values
y = dataset['Species'].values
# Alternative way of selecting features and labels arrays:
# X = dataset.iloc[:, 1:5].values
# y = dataset.iloc[:, 5].values

# Encode Labels to transform into numbers.
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=5)

# Visualize the parallel coordinates
plt.figure(figsize=(15, 10))
parallel_coordinates(dataset.drop("Id", axis=1), "Species")
plt.title('Parallel Coordinates Plot', fontsize=20, fontweight='bold')
plt.xlabel('Features', fontsize=15)
plt.ylabel('Features values', fontsize=15)
plt.legend(loc=1, prop={'size': 15}, frameon=True, shadow=True, facecolor="white", edgecolor="black")
plt.show()

# Andrews Curves
plt.figure(figsize=(15, 10))
andrews_curves(dataset.drop("Id", axis=1), "Species")
plt.title('Andrews Curves Plot', fontsize=20, fontweight='bold')
plt.legend(loc=1, prop={'size': 15}, frameon=True, shadow=True, facecolor="white", edgecolor="black")
plt.show()

# Pairplot
plt.figure()
sns.pairplot(dataset.drop("Id", axis=1), hue="Species", height=3, markers=["o", "s", "D"])
plt.show()

# Boxplot
plt.figure()
dataset.drop("Id", axis=1).boxplot(by="Species", figsize=(15, 10))
plt.show()

# Three dimensional visualization
fig = plt.figure(1, figsize=(20, 15))
ax = Axes3D(fig, elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=X[:, 3] * 50)

for name, label in [('Virginica', 0), ('Setosa', 1), ('Versicolour', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean(),
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'), size=25)

ax.set_title("3D visualization", fontsize=40)
ax.set_xlabel("Sepal Length [cm]", fontsize=25)
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Sepal Width [cm]", fontsize=25)
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Petal Length [cm]", fontsize=25)
ax.w_zaxis.set_ticklabels([])

plt.show()

# Make predictions
# Fitting classifier to the Training set
# Loading libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Instantiate learning model (k = 3)
classifier = KNeighborsClassifier(n_neighbors=3)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluate predictions
cm = confusion_matrix(y_test, y_pred)
cm

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
print('\n------------------------------------------------------------------------\n')

# Using cross-validation for parameter tuning by creating list of K for KNN
k_list = list(range(1, 50, 2))
# creating list of cv scores
cv_scores = []

# perform 10-fold cross validation
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Changing to mis-classification error
MSE = [1 - x for x in cv_scores]

plt.figure()
plt.figure(figsize=(15, 10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)
plt.show()

# finding best k
best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, logreg.predict(X_test)))
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X, y)

# make a prediction for an example of an out-of-sample observation
print(knn.predict([[6, 3, 4, 2]]))