#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html

#1.Harris Corner Detection
#We will understand the concepts behind Harris Corner Detection.
#We will see the functions: cv2.cornerHarris(), cv2.cornerSubPix()

import cv2
import numpy as np

filename = 'j.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

#Corner with SubPixel Accuracy
import cv2
import numpy as np

filename = 'j.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

#cv2.imwrite('subpixel5.png',img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Shi-Tomasi Corner Detector & Good Features to Track=============================
#We will learn about the another corner detector: Shi-Tomasi Corner Detector
#We will see the function: cv2.goodFeaturesToTrack()

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('simple.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10) #try to find 25 best corners
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()


#Introduction to SIFT (Scale-Invariant Feature Transform)
#We will learn about the concepts of SIFT algorithm
#We will learn to find SIFT Keypoints and Descriptors.
#https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/

import cv2
import numpy as np
from time import time
import os

img = cv2.imread('messi.jpg')

if not os.path.exists('messi.jpg'):
	print('Incorrect path or file missing')
	exit(0)

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

tic = time()
sift = cv2.xfeatures2d.SIFT_create()
#kp = sift.detect(gray,None)
kp, des = sift.detectAndCompute(gray,None)

#cv2.drawKeypoints(gray,kp,img)
cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

toc = time()-tic

cv2.imwrite('sift_keypoints.jpg',img)
print('Time: ', toc)
print('Len(kp): ', len(kp))
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Introduction to SURF (Speeded-Up Robust Features)================================================
#We will see the basics of SURF
#We will see SURF functionalities in OpenCV
import cv2
import numpy as np
from time import time
import os
import matplotlib.pyplot as plt

img = cv2.imread('messi.jpg')

if not os.path.exists('messi.jpg'):
	print('Incorrect path or file missing')
	exit(0)

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

tic = time()
#set Hessian Threshold to 4000
surf = cv2.xfeatures2d.SURF_create(4000) 
# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(gray,None)
img2 = cv2.drawKeypoints(gray,kp,None,(255,0,0),4)
toc = time()-tic


print('Time: ', toc)
print('Len(kp): ', len(kp))
print('descriptorSize', surf.descriptorSize())
print('des', des.shape)

plt.imshow(img2),plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()


#FAST Algorithm for Corner Detection============================================================================
#We will understand the basics of FAST algorithm
#We will find corners using OpenCV functionalities for FAST algorithm.
#https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/featureDetection/fast.py
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('simple.jpg',0)
img2 = img.copy()
img3 = img.copy()

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(threshold=50)

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Print all default params
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))

cv2.imwrite('fast_true.png',img2)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)

print("Total Keypoints without nonmaxSuppression: ", len(kp))

img3 = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('fast_false.png',img3)

cv2.imshow('Enable nonmaxSuppression', img2)
cv2.imshow('Disable nonmaxSuppression', img3)
cv2.waitKey(0) 
cv2.destroyAllWindows()

#BRIEF (Binary Robust Independent Elementary Features)===========================================================
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_brief/py_brief.html
import numpy as np
import cv2
from matplotlib import pyplot as plt
from time import time

img = cv2.imread('simple.jpg',0)

tic = time()
# Initiate STAR detector
star = cv2.xfeatures2d.StarDetector_create()

# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# find the keypoints with STAR
kp = star.detect(img,None)

# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

#img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
img2 = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

toc = time()-tic

print('Time: ', toc)
#print(brief.getInt())
print('des-len: ', des.shape)

cv2.imshow('BRIEF', img2)
cv2.waitKey(0) 
cv2.destroyAllWindows()


#ORB (Oriented FAST and Rotated BRIEF)=========================================================================
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
#https://stackoverflow.com/questions/32702433/opencv-orb-detector-finds-very-few-keypoints
import cv2
from matplotlib import pyplot as plt
from time import time

img = cv2.imread('simple.jpg',0)

tic = time()
# Initiate STAR detector
orb = cv2.ORB_create(nfeatures=50, scoreType=cv2.ORB_FAST_SCORE) #cv2.ORB_HARRIS_SCORE, cv2.ORB_FAST_SCORE 

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, None, flags=cv2.DrawMatchesFlags_DEFAULT)
#img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=cv2.DrawMatchesFlags_DEFAULT)
#img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
toc = time()-tic
print('Time: ', toc)

print('des-len: ', des.shape)

#plt.imshow(img2),plt.show()
cv2.imshow('OBR', img2)
cv2.waitKey(0) 
cv2.destroyAllWindows()


#Feature Matching======================================================================================
#We will see how to match features in one image with others.
#We will use the Brute-Force matcher and FLANN Matcher in OpenCV

#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
#https://www.programcreek.com/python/example/89444/cv2.drawMatches

#1.Brute-Force Matching with ORB Descriptors
import cv2
from matplotlib import pyplot as plt
from time import time

#img1 = cv2.imread('messi_face.jpg',0)          # queryImage
img2 = cv2.imread('messi.jpg',0) # trainImage
img1 = img2[47:153, 156:239]

tic = time()

# Initiate ORBT detector
orb = cv2.ORB_create() #cv2.ORB_HARRIS_SCORE, cv2.ORB_FAST_SCORE; nfeatures=50, scoreType=cv2.ORB_FAST_SCORE 

# compute the descriptors with ORB
#kp1 = orb.detect(img1,None)
#kp1, des1 = orb.compute(img1, kp1)
#kp2 = orb.detect(img2,None)
#kp2, des2 = orb.compute(img2, kp2)
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

toc = time()-tic
print('Time: ', toc)

print('des1-len1: ', des1.shape)
print('des2-len2: ', des2.shape)
print('Matches: ', len(matches))

img4 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DrawMatchesFlags_DEFAULT)
img5 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DrawMatchesFlags_DEFAULT)


plt.imshow(img3),plt.show()

#cv2.imshow('OBR', img3)
#cv2.waitKey(0) 
#cv2.destroyAllWindows()

#2.Brute-Force Matching with SIFT Descriptors and Ratio Test===========================================================
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
#https://www.programcreek.com/python/example/89444/cv2.drawMatches

#1.Brute-Force Matching with ORB Descriptors
import cv2
from matplotlib import pyplot as plt
from time import time

#img1 = cv2.imread('messi_face.jpg',0)          # queryImage
img2 = cv2.imread('messi.jpg',0) # trainImage
img1 = img2[47:153, 156:239]     # queryImage

tic = time()

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# compute the descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Match descriptors.
matches = bf.match(des1,des2)

# Apply ratio test
good = []
for i, m in enumerate(matches):
	if i < len(matches) - 1 and m.distance < 0.7*matches[i+1].distance:
		good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

toc = time()-tic
print('Time: ', toc)

print('des1-len1: ', des1.shape)
print('des2-len2: ', des2.shape)
print('Matches: ', len(matches))
print(type(matches))
print(enumerate(matches))
print('len(good): ', len(good))

img4 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DrawMatchesFlags_DEFAULT)
img5 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DrawMatchesFlags_DEFAULT)


plt.imshow(img3),plt.show()

#cv2.imshow('OBR', img3)
#cv2.waitKey(0) 
#cv2.destroyAllWindows()


#3.FLANN based Matcher======================================================================================================
import cv2
from matplotlib import pyplot as plt
from time import time

#img1 = cv2.imread('messi_face.jpg',0)          # queryImage
img2 = cv2.imread('messi.jpg',0) # trainImage
img1 = img2[47:153, 156:239]     # queryImage

tic = time()

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# compute the descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

toc = time()-tic
print('Time: ', toc)

print('des1-len1: ', des1.shape)
print('des2-len2: ', des2.shape)
print('Matches: ', len(matches))
print(type(matches))
print(enumerate(matches))

img4 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DrawMatchesFlags_DEFAULT)
img5 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DrawMatchesFlags_DEFAULT)


plt.imshow(img3),plt.show()

#cv2.imshow('OBR', img3)
#cv2.waitKey(0) 
#cv2.destroyAllWindows()

#4.Feature Matching + Homography to find Objects============================================================================
#We will mix up the feature matching and findHomography from calib3d module to find known objects in a complex image.
import cv2
from matplotlib import pyplot as plt
from time import time
import numpy as np

#img1 = cv2.imread('messi_face.jpg',0)          # queryImage
img2 = cv2.imread('messi.jpg',0) # trainImage
img1 = img2[47:153, 156:239]     # queryImage

tic = time()

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# compute the descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

MIN_MATCH_COUNT = 10
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,1, cv2.LINE_AA)

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

toc = time()-tic
print('Time: ', toc)

print('des1-len1: ', des1.shape)
print('des2-len2: ', des2.shape)
print('Matches: ', len(matches))
print(type(matches))
print(enumerate(matches))

img4 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DrawMatchesFlags_DEFAULT)
img5 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DrawMatchesFlags_DEFAULT)


plt.imshow(img3, 'gray'),plt.show()

#cv2.imshow('OBR', img3)
#cv2.waitKey(0) 
#cv2.destroyAllWindows()