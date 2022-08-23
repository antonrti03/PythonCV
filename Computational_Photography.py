#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_table_of_contents_photo/py_table_of_contents_photo.html
#Here you will learn different OpenCV functionalities related to Computational Photography like image denoising etc.

#1. Image Denoising====================================================
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html#non-local-means
#You will learn about Non-local Means Denoising algorithm to remove noise in the image.
#You will see different functions like cv2.fastNlMeansDenoising(), cv2.fastNlMeansDenoisingColored() etc.

#1.1 cv2.fastNlMeansDenoisingColored()
import numpy as np
import cv2
from matplotlib import pyplot as plt
from time import time

img = cv2.imread('opencv_logo.png')
tic = time()
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
toc = time()-tic
print('Time: ', toc)
plt.subplot(121), plt.title('Original'), plt.imshow(img)
plt.subplot(122), plt.title('Denoising'), plt.imshow(dst)
plt.show()

#1.2. cv2.fastNlMeansDenoisingMulti()
import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('vtest.avi')

# create a list of first 5 frames
img = [cap.read()[1] for i in range(5)]

# convert all to grayscale
gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]

# convert all to float64
gray = [np.float64(i) for i in gray]

# create a noise of variance 25
noise = np.random.randn(*gray[1].shape)*10

# Add this noise to images
noisy = [i+noise for i in gray]

# Convert back to uint8
noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]

# Denoise 3rd frame considering all the 5 frames
dst = cv2.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)

plt.subplot(131),plt.imshow(gray[2],'gray')
plt.subplot(132),plt.imshow(noisy[2],'gray')
plt.subplot(133),plt.imshow(dst,'gray')
plt.show()


#2.Image Inpainting=========================================================================
import numpy as np
import cv2

img = cv2.imread('messi_2.jpg')
mask = cv2.imread('mask2.png',0)

dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()