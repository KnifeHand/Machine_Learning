"""
Author: Matt Hurt
Machine Learning - 3120: Final Project
Created: April 25, 2021
"""
# todo: try classification using one-dimensional data clustering of radii
# http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf

# import classifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import math
import numpy as np
import argparse
import glob
import cv2

# construct argument parser and parse arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to image")
# args = vars(ap.parse_args())

# image = cv2.imread(args["image"])
image = cv2.imread("read_image/nickel.jpg")

# resize image while retaining aspect ratio
d = 1024 / image.shape[1]
dim = (1024, int(image.shape[0] * d))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# create a copy of the image to display results
output = image.copy()

# convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# improve contrast accounting for differences in lighting conditions:
# create a CLAHE object to apply contrast limiting adaptive histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)


def calcHistogram(img):
    # create mask
    m = np.zeros(img.shape[:2], dtype="uint8")
    (w, h) = (int(img.shape[1] / 2), int(img.shape[0] / 2))
    cv2.circle(m, (w, h), 60, 255, -1)

    # calcHist expects a list of images, color channels, mask, bins, ranges
    h = cv2.calcHist([img], [0, 1, 2], m, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # return normalized "flattened" histogram
    return cv2.normalize(h, h).flatten()


def calcHistFromFile(file):
    img = cv2.imread(file)
    return calcHistogram(img)


# define Enum class
class Enum(tuple): __getattr__ = tuple.index


# Enumerate material types for use in classifier
Material = Enum(('Penny', 'Nickel', 'Quarter', 'Dime'))

# locate sample image files
sample_images_penny = glob.glob("sample_images/penny/*")
sample_images_dime = glob.glob("sample_images/dime/*")
sample_images_nickel = glob.glob("sample_images/nickel/*")
sample_images_quarter = glob.glob("sample_images/quarter/*")

# define training data and labels
X = []
y = []

# compute and store training data and labels
for i in sample_images_penny:
    X.append(calcHistFromFile(i))
    y.append(Material.Penny)
for i in sample_images_nickel:
    X.append(calcHistFromFile(i))
    y.append(Material.Nickel)
for i in sample_images_quarter:
    X.append(calcHistFromFile(i))
    y.append(Material.Quarter)
for i in sample_images_dime:
    X.append(calcHistFromFile(i))
    y.append(Material.Dime)

# instantiate classifier
# Multi-layer Perceptron
# score: 0.974137931034
clf = MLPClassifier(solver="lbfgs")

# split samples into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2)

print("X_train = ", X_train)
print("X_test = ", X_test)
print("y_train = ", y_train)
print("y_test = ", y_test)

# train and score classifier
clf.fit(X_train, y_train)
score = int(clf.score(X_test, y_test) * 100)
print("Classifier mean accuracy: ", score)

# blur the image using Gaussian blurring, where pixels closer to the center
# contribute more "weight" to the average, first argument is the source image,
# second argument is kernel size, third one is sigma (0 for autodetect)
# we use a 7x7 kernel and let OpenCV detect sigma
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# circles: A vector that stores x, y, r for each detected circle.
# src_gray: Input image (grayscale)
# CV_HOUGH_GRADIENT: Defines the detection method.
# dp = 2.2: The inverse ratio of resolution
# min_dist = 100: Minimum distance between detected centers
# param_1 = 200: Upper threshold for the internal Canny edge detector
# param_2 = 100*: Threshold for center detection.
# min_radius = 50: Minimum radius to be detected.
# max_radius = 120: Maximum radius to be detected.
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=2.2, minDist=100,
                           param1=200, param2=100, minRadius=50, maxRadius=120)


def predictMaterial(roi):
    # calculate feature vector for region of interest
    hist = calcHistogram(roi)

    # predict material type
    s = clf.predict([hist])

    # return predicted material type
    return Material[int(s)]


# todo: refactor
diameter = []
materials = []
coordinates = []

count = 0
if circles is not None:
    # append radius to list of diameters (we don't bother to multiply by 2)
    for (x, y, r) in circles[0, :]:
        diameter.append(r)

    # convert coordinates and radii to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over coordinates and radii of the circles
    for (x, y, d) in circles:
        count += 1

        # add coordinates to list
        coordinates.append((x, y))

        # extract region of interest
        roi = image[y - d:y + d, x - d:x + d]

        # try recognition of material type and add result to list
        material = predictMaterial(roi)
        materials.append(material)

        # write masked coin to file
        if False:
            m = np.zeros(roi.shape[:2], dtype="uint8")
            w = int(roi.shape[1] / 2)
            h = int(roi.shape[0] / 2)
            cv2.circle(m, (w, h), d, (255), -1)
            maskedCoin = cv2.bitwise_and(roi, roi, mask=m)
            cv2.imwrite("extracted/01coin{}.png".format(count), maskedCoin)

        # draw contour and results in the output image
        cv2.circle(output, (x, y), d, (0, 255, 0), 2)
        cv2.putText(output, material,
                    (x - 40, y), cv2.FONT_HERSHEY_PLAIN,
                    1.5, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

# print("roi = ", roi)
print("list of diameter = ", diameter)

# get biggest diameter
biggest = max(diameter)
i = diameter.index(biggest)
print("biggest value", biggest)
print("x = ", x)

# scale everything according to maximum diameter
# todo: this should be chosen by the user
if materials[i] == "Dime":
    diameter = [x / biggest * 17.91 for x in diameter]
    scaledTo = "Scaled to 10 cents"
elif materials[i] == "Nickel":
    diameter = [x / biggest * 21.21 for x in diameter]
    scaledTo = "Scaled to 5 cents"
elif materials[i] == "Quarter":
    diameter = [x / biggest * 24.26 for x in diameter]
    scaledTo = "Scaled to 25 cents"
elif materials[i] == "Penny":
    diameter = [x / biggest * 19.05 for x in diameter]
    scaledTo = "Scaled to 1 cent"
else:
    scaledTo = "unable to scale.."

print("diameter", diameter)

i = 0
total = 0
while i < len(diameter):
    d = diameter[i]
    m = materials[i]
    (x, y) = coordinates[i]
    t = "Unknown"

    # compare to known diameters with some margin for error
    if math.isclose(d, 17.91, abs_tol=0) and m == "Dime":
        t = "10 cents"
        total += 10
    elif math.isclose(d, 24.26, abs_tol=0) and m == "Quarter":
        t = "25 cents"
        total += 25
    elif math.isclose(d, 21.21, abs_tol=0) and m == "Nickel":
        t = "5 cents"
        total += 5
    elif math.isclose(d, 19.05, abs_tol=0) and m == "Penny":
        t = "1 cent"
        total += 1

    # write result on output image
    cv2.putText(output, t,
                (x - 40, y + 22), cv2.FONT_HERSHEY_PLAIN,
                1.5, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    i += 1

    print("m = ", m)
    print("d = ", d)

# resize output image while retaining aspect ratio
d = 768 / output.shape[1]
dim = (768, int(output.shape[0] * d))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA)

# write summary on output image
cv2.putText(output, scaledTo,
            (5, output.shape[0] - 40), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), lineType=cv2.LINE_AA)
cv2.putText(output, "Coins detected: {}, US {:2}".format(count, total / 100),
            (5, output.shape[0] - 24), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), lineType=cv2.LINE_AA)
cv2.putText(output, "Classifier mean accuracy: {}%".format(score),
            (5, output.shape[0] - 8), cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), lineType=cv2.LINE_AA)

# show output and wait for key to terminate program
cv2.imshow("Output", np.hstack([image, output]))
cv2.waitKey(0)
