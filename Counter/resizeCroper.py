import cv2 as cv
import glob
import os
import numpy as np


inputFolder = "ImagesDir"
folderLen = len(inputFolder)
# os.mkdir('Resized')
for img in glob.glob(inputFolder + "/*.jpg"):
    image = cv.imread(img, cv.IMREAD_COLOR)
    imgResized = cv.resize(image, (150, 150))
    # src = cv.imread(imgResized, cv.IMREAD_COLOR)
    # src = cv.cvtColor(imgResized, cv.COLOR_BGR2RGB)
    #
    # # Find Circles
    # gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # gray = cv.medianBlur(gray, 5)
    # rows = gray.shape[0]
    # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows, param1=50, param2=30, minRadius=0, maxRadius=0)
    #
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     width = 15
    #     RGBcolor = (0, 255, 68)
    #
    #     for pt in circles[0, :]:  # pt is a point (a, b, r) which represents a circle of radius r with center (a, b).
    #         # Hopefully only 1 circle is in the circles list for your
    #         center = (pt[0], pt[1])
    #         radius = pt[2]
    #         # circle center
    #         cv.circle(src, center, 1, (0, 100, 100), 3)
    #         # circle outline
    #         srcCopy = np.copy(src)
    #         cv.circle(srcCopy, center, radius, RGBcolor, width)  # Don't draw circle if you just want to crop is
    #
    #         # Crop based off center-point and radius.
    #         a, b = center
    #         r = radius
    #         cropImg = src[b - r:b + r, a - r:a + r, :]


            # plt.imshow(cropImg)
            # plt.title("Cropped Image")
            # plt.show()
            # cv2.imshow("coin", cropImg)
            # cv2.waitKey(0)
            # return cropImg

    cv.imwrite("Resized" + img[folderLen:], imgResized)

print("Images have been cropped....")
