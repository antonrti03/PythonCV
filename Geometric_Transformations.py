#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
#Learn to apply different geometric transformation to images like translation, rotation, affine transformation etc.
#You will see these functions: cv2.getPerspectiveTransform

#Scaling===============
import cv2
import numpy as np

img = cv2.imread('beans.jpg')

e1 = cv2.getTickCount()

res = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC) #cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC

#OR

height, width = img.shape[:2]
res2 = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

e2 = cv2.getTickCount()
t = (e2 - e1)/cv2.getTickFrequency()
print('Processed time: ', t)

cv2.imshow('res', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Translation==========================
import cv2
import numpy as np

img = cv2.imread('beans.jpg',0)
rows,cols = img.shape

M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img, M, (cols,rows)) #the size of the output image, which should be in the form of (width, height)

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Rotation==========================
import cv2
import numpy as np

img = cv2.imread('beans.jpg',0)
rows,cols = img.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2), 90, 1)
dst = cv2.warpAffine(img, M, (cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Affine Transformation==========================
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('beans.jpg',0)
rows,cols = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()


#Perspective Transformation==========================
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('beans.jpg',0)
rows,cols = img.shape

pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()