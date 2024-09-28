import cv2 as cv
import numpy as np

#image transformations

img = cv.imread(r"C:\Users\ASUS\Pictures\Screenshot_2022_1126_205325.png")
cv.imshow('poojitha',img)

#translation
def translate(img,x,y):#x,y no pixels
    transmat=np.float32([[1,0,x],[0,1,y]])
    dimensions=(img.shape[1],img.shape[0])
    return cv.warpAffine(img,transmat,dimensions)
    
#-x-->left
#-y-->up
#x-->right
#y-->down
translated=translate(img,100,-100)
cv.imshow("tranlated image",translated)

def rotate(img,angle,rotPoint=None):
    height,width=img.shape[:2]

    if(rotPoint is None):
        rotPoint==(width//2,height//2)
    rotMat=cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions=(height,width)
    return cv.warpAffine(img,rotMat,dimensions)
rotation=rotate(img,45)
cv.imshow('roateted',rotation)

resized=cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow("resized",resized)
        
#flip
flip=cv.flip(img,1)
cv.imshow("Flip",flip)
crp=flip[50:200,200:400]
cv.imshow("croppp",flip)
