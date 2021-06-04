#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 00:58:52 2021

@author: harshraj
"""

import cv2 
from skimage import io 
from matplotlib import pyplot as plt
import numpy as np
import pywt
import pywt.data
import matplotlib.image as mpimg

img = cv2.imread("/Users/harshraj/3.png", 1)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
lab_img=cv2.cvtColor(img ,cv2.COLOR_BGR2LAB)
l , a, b= cv2.split(lab_img)
equ= cv2.equalizeHist(l)
# for image 1.PNG the pixel are dived
# from 100 to 230 approx thats why i have to yake the range
# 141 to 255 if i take 0 to 255 it came all white 
ret, thresold =cv2.threshold(img_gray,130,255,cv2.THRESH_BINARY)
kernal=np.ones((2,2), np.uint8)

dilation=cv2.dilate(thresold ,kernal ,iterations=0)
erosion=cv2.erode(thresold ,kernal ,iterations=2)
opening=cv2.morphologyEx(thresold ,cv2.MORPH_OPEN ,kernal )
closing=cv2.morphologyEx(thresold ,cv2.MORPH_CLOSE,kernal )
boundary=cv2.morphologyEx(thresold ,cv2.MORPH_GRADIENT, kernal )
top_hat=cv2.morphologyEx(thresold ,cv2.MORPH_TOPHAT, kernal )
bottom_hat=cv2.morphologyEx(thresold ,cv2.MORPH_BLACKHAT, kernal )
# wavelate transform


coeffs2=pywt.dwt2(equ, 'haar' , mode='periodization')

cA, (cH , cV, cD) = coeffs2
imgr=pywt.idwt2(coeffs2, 'haar' , mode='periodization')
imgr=np.uint8(imgr)

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)

plt.imshow(cA, cmap='gray')
plt.title('cA: approx cofficient.' , fontsize=10)
plt.subplot(2,2,2)

plt.imshow(cH, cmap=plt.cm.gray)
plt.title('cH: HORIZONTAL Datailed cofficient.' , fontsize=10)

plt.subplot(2,2,3)

plt.imshow(cV, cmap=plt.cm.gray)
plt.title('cV: VERTICAL Datailed cofficient.' , fontsize=10)

plt.subplot(2,2,4)

plt.imshow(cD, cmap=plt.cm.gray)
plt.title('cD: DIAGONAL Datailed cofficient.' , fontsize=10)

plt.show()

plt.imshow (imgr,cmap=plt.cm.gray)
plt.title ('reconstruction of image ', fontsize=10)






#original image histogram
 
plt.subplot(211)
plt.hist(l.flat,bins=255 , range=(0,255))
plt.title('image histogram')

#IMAGE histogram equalization
 
plt.subplot(212)
plt.hist(equ.flat,bins=255 , range=(0,255))
plt.title('image histogram equlization', fontsize=10)


plt.show()


cv2.imshow("histogram equalized image ", equ)
cv2.imshow("thresold image", thresold)
cv2.imshow("dilated image", dilation)
cv2.imshow("erosion image", erosion)
cv2.imshow("opening  image", opening)
cv2.imshow("closing  image", closing)
cv2.imshow("boundry detection",boundary)
cv2.imshow("top hat iamge",top_hat)
cv2.imshow("bottom HAT",bottom_hat)
cv2.waitKey(10)
cv2.destroyAllWindows()


