

import cv2 #package

#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True,
			#help = "Path to the image")
#args = vars(ap.parse_args())
#image = cv2.imread(args["image"])   #đọc file ảnh thành một mảng NumPy, mảng NumPy là image
image = cv2.imread("d:/Python_OpenCV/CV/Lena.bmp")
print ("width: %d pixels" % (image.shape[1]))  #số pixel theo chiều X
print ("height: %d pixels" % (image.shape[0])) #là số pixel theo chiều Y
print ("channels: %d channels" % (image.shape[2])) #số channels
#
cv2.imshow("Image", image)  #hiển thị bức ảnh trên một window mới, với 2 đối số là tên của window và tên của mảng NumPy muốn hiển thị.
cv2.imwrite("new.jpg", image) #lưu mảng NumPy thành một file ảnh mới
cv2.waitKey(0)
#Chạy đoạn code trên như sau:
# % python getinfo.py -i obama_fun.jpg


#------------video input-------------
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
	ret,frame = cap.read()
	cv2.rectangle(frame, (100, 100), (200, 200), [255, 0, 0], 2)
    # Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows() 


# ------------imread()--------------
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('D:\\Python_OpenCV\\CV\\lena512.bmp', 0) #Màu là 1, grayscale là 0, và không thay đổi là -1
#img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)

cv2.imshow('image',img) #cv2.imshow (title, image) để hiển thị hình ảnh
cv2.imwrite("new.jpg", img)

cv2.waitKey(0) #cv2.waitKey (0) để chờ cho đến khi bất kỳ phím nào được nhấn
cv2.destroyAllWindows()  #cv2.destroyAllWindows () để đóng tất cả
print('W: ', img.shape[0])
print('H: ', img.shape[1])

print(ord('q'))


#----------matplotlib--------------
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('D:\\Python_OpenCV\\CV\\abc.png', 0) #Màu là 1, grayscale là 0, và không thay đổi là -1
#img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
plt.show()
#
cv2.imshow('image',img) 
cv2.imwrite("new.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
print('W: ', img.shape[0])
print('H: ', img.shape[1])
print(ord('q'))


#----------------------imdilate()---------------
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena512.bmp', 0)
kernel = np.ones((3, 3), np.uint8)
img = cv2.dilate(img, kernel, iterations=1)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#print(img)
print(img.shape)
print(img.shape[0])
print(img.shape[1])

#-------open an image from Path
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

#import convolute_lib as cnn

img_path = str(Path(__file__).parent.parent / 'D:/Python_OpenCV/CV/beans.jpg')

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

cv2.imshow('filled', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
#-----------------------------------------------------------