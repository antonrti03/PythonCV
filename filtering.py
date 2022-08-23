#https://techmaster.vn/posts/35477/convolution-xu-ly-anh-qua-vi-du-python-thuc-te
#https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
# cv2.filter2D(img, -1, filter[1])
# cv2.Canny()
# for deep learning: Contour detection, Clustering, Noise remove, Image Transform (rotate, zoom, ...)
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
#import convolute_lib as cnn

img_path = str(Path(__file__).parent.parent / 'D:/Python_OpenCV/CV/beans.jpg')

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

identity = np.array((
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]))

edge = np.array((
    [0,  1,  0],
    [1, -4,  1],
    [0,  1,  0]))

boxblur = (1.0 / 9) * np.array(
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]])

gaussian = (1.0 / 16) * np.array(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]])

emboss = np.array(
    [[-2, -1,  0],
     [-1,  1,  1],
     [ 0,  1,  2]])

square = np.array(
    [[ 0,  2,  0],
     [-2, -1,  2],
     [ 0, -2,  0]])

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
    [0, -2, 0],
    [-2, 10, -2],
    [0, -2, 0]))

laplacian = (1.0 / 16) * np.array(
    [[ 0,  0, -1,  0,  0],
     [ 0, -1, -2, -1,  0],
     [-1, -2, 16, -2, -1],
     [ 0, -1, -2, -1,  0],
     [ 0,  0, -1,  0,  0]])

sobelLeft = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]))

sobelRight = np.array((
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]))

sobelTop = np.array((
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]))

sobelBottom = np.array((
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]))

filters = [
    ("Identity", identity),
    ("Edge", edge),
    ("Box Blur", boxblur),
    ("Square", square),
    ("Gaussian", gaussian),
    ("Emboss", emboss),
    ("Small blur", smallBlur),
    ("Large blur", largeBlur),
    ("Sharpen", sharpen),
    ("Laplacian", laplacian),
    ('Sobel Left', sobelLeft),
    ('Sobel Right', sobelRight),
    ('Sobel Top', sobelTop),
    ('Sobel Bottom', sobelBottom)
]

fig = plt.figure(figsize=(12, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.1)

for i, filter in enumerate(filters):
    axes = fig.add_subplot(3, 5, i+1)
    axes.set(title=filter[0])
    axes.grid(False)
    axes.set_xticks([])
    axes.set_yticks([])
    #img_out = cnn.convolve_np4(img, filter[1])
    img_out = cv2.filter2D(img, -1, filter[1])
    axes.imshow(img_out, cmap='gray', vmin=0, vmax=255)


# Canny detector
axes = fig.add_subplot(3, 5, 15)
axes.set(title='Canny')
axes.grid(False)
axes.set_xticks([])
axes.set_yticks([])
img_out = cv2.Canny(img, 150, 200)
axes.imshow(img_out, cmap='gray', vmin=0, vmax=255)

plt.show()