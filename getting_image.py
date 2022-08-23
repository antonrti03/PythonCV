import cv2

# Load an color image in grayscale
img = cv2.imread('beans.jpg',0)
cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE) #cv2.WINDOW_AUTOSIZE, cv2.WINDOW_NORMAL
cv2.imshow('image',img)
#cv2.imwrite('messigray.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#----------------------------
#Below program loads an image in grayscale, displays it, save the image if you press ‘s’ and exit, 
#or simply exit without saving if you press ESC key.
import numpy as np
import cv2

img = cv2.imread('beans.jpg',0)
cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()



# Using Matplotlib----------------------
# You can zoom images, save it etc using Matplotlib.
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('beans.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()