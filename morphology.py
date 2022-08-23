#https://www.stdio.vn/articles/xu-ly-anh-voi-opencv-cac-phep-toan-hinh-thai-hoc-421
#Morphology:
#Dilation gọi là D(i): giãn nở.
#Erosion gọi là E(i): co.
#Một chu trình E(i)-D(i) gọi là phép mở (Opening).
#Một chu trình D(i)-E(i) gọi là phép đóng (Closing).

#Application:
#Trích lọc biên ảnh (Boundary extraction).
#Tô đầy vùng (Region fill).
#Trích lọc các thành phần liên thông (Extracting connected components).
#Làm mỏng đối tượng trong ảnh (Thinning).
#Làm dày đối tượng trong ảnh (Thickening).
#Tìm xương đối tượng trong ảnh (Skeletons).
#Cắt tỉa đối tượng trong ảnh (Pruning).

from cv2 import cv2
import numpy as np
#from skimage.morphology import opening
 
img = cv2.imread("beans.jpg") 
kernel = np.ones((5,5),np.uint8)  #matrix '1' of size 5x5
 
# Bước 1: Chuyển về ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray', gray)

# Bước 2: Làm mờ ảnh
blur = cv2.GaussianBlur(gray, (9, 9), 1)  # '1' matrix 9x9, sigma=1
#cv2.imshow('GaussianBlur', blur)
 
# Bước 3: Lọc nhiễu
new = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, -5)
#cv2.imshow('adaptiveThreshold', new) 

# Bước 4: Opening
opening = cv2.morphologyEx(new, cv2.MORPH_OPEN, kernel) #MORPH_ERODE, MORPH_DILATE, MORPH_CLOSE
cv2.imshow('Opening', opening)

closing = cv2.morphologyEx(new, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closing', closing)

erode = cv2.morphologyEx(new, cv2.MORPH_ERODE, kernel, iterations = 1)
cv2.imshow('erode', erode)

dilate = cv2.morphologyEx(new, cv2.MORPH_DILATE, kernel)
cv2.imshow('dilate', dilate)

#Kết thúc chương trình với:
cv2.waitKey(0)
cv2.destroyAllWindows()