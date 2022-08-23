#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_optimization/py_optimization.html

#=======================Basic operations=================================
import cv2
# Load an color image in grayscale
img = cv2.imread('beans.jpg',1)

px = img[100,100]
print(px)

# accessing only blue pixel
blue = img[100,100,0]
print(blue)

#You can modify the pixel values the same way
img[100,100] = [255,255,255]
print(img[100,100])

#Accessing Image Properties
print('WxHxC:', img.shape)
print(img.size)
print(img.dtype)

#Image ROI
#ball = img[280:340, 330:390]
#img[273:333, 100:160] = ball

#Splitting and Merging Image Channels
b,g,r = cv2.split(img) # b = img[:,:,0], g = img[:,0,:], r = img[0,:,:]
img = cv2.merge((b,g,r))

#Making Borders for Images (Padding): cv2.copyMakeBorder() function

#====================================
cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE) #cv2.WINDOW_AUTOSIZE, cv2.WINDOW_NORMAL
cv2.imshow('image',img)
#cv2.imwrite('messigray.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#==============================Arithmetic Operations on Images==================================
# You will learn these functions : cv2.add(), cv2.addWeighted() etc.

#Image Addition=============
import numpy as np
import cv2

x = np.uint8([250])
y = np.uint8([10])
print(cv2.add(x,y)) #250+10 = 260 => 255
print(x+y)          #250+10 = 260 % 256 = 4

#Image Blending==============
import numpy as np
import cv2

img1 = cv2.imread('beans.jpg')
img2 = cv2.imread('abc.png')

dst = cv2.addWeighted(img1,0.7, img2,0.3, 0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Bitwise Operations===========
import numpy as np
import cv2

# Load two images
img1 = cv2.imread('abc.png')
img2 = cv2.imread('beans.jpg')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()


#============Performance Measurement and Improvement Techniques======================
#You will see these functions : cv2.getTickCount, cv2.getTickFrequency etc

#Measuring Performance with OpenCV
import numpy as np
import cv2

img1 = cv2.imread('beans.jpg')

e1 = cv2.getTickCount()  # ~ time.time()

for i in range(5,49,2):
	img1 = cv2.medianBlur(img1,i)

e2 = cv2.getTickCount()
t = (e2 - e1)/cv2.getTickFrequency()

print(t)

#cv2.useOptimized()======
import numpy as np
import cv2

img1 = cv2.imread('beans.jpg')

cv2.useOptimized()

e1 = cv2.getTickCount()  # ~ time.time()

for i in range(5,49,2):
	img1 = cv2.medianBlur(img1,i)

e2 = cv2.getTickCount()
t = (e2 - e1)/cv2.getTickFrequency()

print(t)