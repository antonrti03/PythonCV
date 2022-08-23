#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html#fourier-transform
#To find the Fourier Transform of images using OpenCV
#To utilize the FFT functions available in Numpy
#Some applications of Fourier Transform
#We will see following functions : cv2.dft(), cv2.idft() etc

#Fourier Transform in Numpy===========================================
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('beans.jpg',0)
f = np.fft.fft2(img) #np.ifft2()
fshift = np.fft.fftshift(f) #np.fft.ifftshift()
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

#Fourier Transform in OpenCV===========================================
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('beans.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()