import numpy as np
import cv2

image1 = cv2.imread('abc.png')
image2 = cv2.imread('anhcat.jpg')

crop1 = image1[100:199, 100:299]
cv2.imshow('crop1', crop1)
crop2 = image2[0:99, 0:199]
cv2.imshow('crop2', crop2)

img3= cv2.add(crop1,crop2) #lệnh ghép 2 ảnh với nhau
cv2.imshow('img3', img3)

Y= (img3.shape[0]) #lấy chiều cao của ảnh
print (Y)
X= (img3.shape[1]) # lấy chiều rộng của ảnh
print (X)

cv2.waitKey(0)
cv2.destroyAllWindows()

#https://kipalog.com/posts/Tim-hieu-xu-ly-anh-bang-OpenCV-trong-Python---Thuc-hanh-2