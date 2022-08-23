#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
#Blur imagess with various low pass filters
#Apply custom-made filters to images (2D convolution)

#2D Convolution ( Image Filtering )==========================
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('beans.jpg')

kernel = np.ones((5,5), np.float32)/25
dst = cv2.filter2D(img,-1, kernel)

plt.subplot(121),plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

cv2.imshow('res', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Image Blurring (Image Smoothing)==========================
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('beans.jpg')

blur = cv2.blur(img,(5,5))

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


#Gaussian Filtering==========================
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('beans.jpg')

blur = cv2.GaussianBlur(img,(5,5),0) #sigma=0

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])


#Median Filtering==========================
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('beans.jpg')

blur = cv2.medianBlur(img,5)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()


#Bilateral Filtering==========================
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('beans.jpg')

blur = cv2.bilateralFilter(img,9,75,75) #to blur edges

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()