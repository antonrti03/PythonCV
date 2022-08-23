#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#contours-getting-started
#Learn to find contours, draw contours etc
#You will see these functions : cv2.findContours(), cv2.drawContours()

import numpy as np
import cv2

img = cv2.imread('beans.jpg',1)

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE

#How to draw the contours?
#To draw all the contours in an image:
img = cv2.drawContours(img, contours, -1, (0,255,0), 0)
#To draw an individual contour, say 4th contour:
#img = cv2.drawContours(img, contours, 3, (0,255,0), 0)
#But most of the time, below method will be useful:
#cnt = contours[4]
#img = cv2.drawContours(img, [cnt], 0, (0,255,0), 0)

cv2.imshow('res', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(type(contours))
print(type(hierarchy))
print(len(contours))

#Contour Features=======================================================
import numpy as np
import cv2

img = cv2.imread('beans.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

#1. Moments
cnt = contours[0]
M = cv2.moments(cnt)
print(len(M))

#2. Contour Area
area = cv2.contourArea(cnt)
print(area)

#3. Contour Perimeter
perimeter = cv2.arcLength(cnt,True)
print(perimeter)

#4. Contour Approximation
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

#7. Bounding Rectangle
#7.a. Straight Bounding Rectangle
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#7.b. Rotated Rectangle
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img = cv2.drawContours(img,[box],0,(0,0,255),2)
#===============================================
cv2.imshow('image',img)
#cv2.imwrite('messigray.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()