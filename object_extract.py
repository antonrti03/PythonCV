import numpy as np
import cv2

img= np.uint8([[[10, 72 , 250]]]) #Đổi 1 điểm ảnh từ hệ BGR sang HSV
hsv_img3 = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
print(hsv_img3) #in giá trị màu trong hệ HSV

img2=cv2.imread("anhcat.jpg",1)
hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV) #Đổi ảnh sang hệ HSV
cv2.imshow ("hien anh HSV", hsv_img2)
#cv2.waitKey(0)

cv2.imwrite('anhHSV.jpg',hsv_img2)
min_mau = np.array([7, 140, 185]) #xác định giá trị nhỏ nhất mảng màu
max_mau = np.array([8, 245, 250]) #xác định giá trị lớn nhất mảng màu
mask = cv2.inRange(hsv_img2, min_mau, max_mau) #tạo một mặt nạ trong khoảng màu min, max
cv2.imshow ("mat na anh", mask)
#cv2.waitKey(0)

#Tạo 1 ảnh mới bằng phép and các bit màu của ảnh với mặt nạ
final = cv2.bitwise_and(img2, img2, mask = mask)
cv2.imshow ("mat na anh", final)
cv2.waitKey(0)
cv2.destroyAllWindows()