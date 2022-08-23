#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html
#We will see these functions: cv2.pyrUp(), cv2.pyrDown()
#There are two kinds of Image Pyramids. 1) Gaussian Pyramid and 2) Laplacian Pyramids

import numpy as np
import cv2

img = cv2.imread('beans.jpg',0)
lower_reso = cv2.pyrDown(img)
higher_reso2 = cv2.pyrUp(lower_reso)

cv2.imshow('res', higher_reso2)
cv2.waitKey(0)
cv2.destroyAllWindows()