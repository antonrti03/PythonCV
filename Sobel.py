from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2

def sobel_filters(img):

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)
    #yield G
    #yield theta

img = cv2.imread('beans.jpg', 0)
#blur = cv2.GaussianBlur(img,(5,5),1)
(edges, theta1) = sobel_filters(img)

#cv2.imshow('image', np.uint8(edges))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.subplot(121), plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.uint8(edges),cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

#uint_img = np.array(edges*255).astype('uint8')
#grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
#print(type(grayImage))
#print(type(uint_img))
#cv2.imshow('image',grayImage)
#cv2.waitKey(0)
#cv2.destroyAllWindows()