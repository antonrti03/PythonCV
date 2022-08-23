#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
# BGR  <-> Gray, BGR <-> HSV
# You will learn following functions : cv2.cvtColor(), cv2.inRange() etc.
import numpy as np
import cv2

img1 = cv2.imread('beans.jpg', 1) # cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2HSV

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)

cv2.imshow('image',img1)
#cv2.imwrite('messigray.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Object Tracking
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()