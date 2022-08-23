from cv2 import cv2
import numpy as np
#from skimage.morphology import opening
 
img = cv2.imread("beans.jpg") 
kernel = np.ones((5,5),np.uint8)  #matrix '1' of size 5x5
 
# Bước 1: Chuyển về ảnh xám
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

# Bước 2: Làm mờ ảnh
blur = cv2.GaussianBlur(gray, (9, 9), 1)  # '1' matrix 9x9, sigma=1
cv2.imshow('GaussianBlur', blur)
 
# Bước 3: Lọc nhiễu
new = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, -5)
 
# Bước 4: Opening
opening = cv2.morphologyEx(new, cv2.MORPH_OPEN, kernel) #MORPH_ERODE, MORPH_DILATE, MORPH_CLOSED
cv2.imshow('Opening', opening)

# Bước 5: Đếm
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
# Kiểm tra kết quả: Draws contours outlines or filled contours
fill = cv2.drawContours(img, contours, -1, (0, 0, 255), 1) #BLUE - GREEN - RED
cv2.imshow('filled', fill)
print("Count: " + str(len(contours)))

#Kết thúc chương trình với:
cv2.waitKey(0)
cv2.destroyAllWindows()