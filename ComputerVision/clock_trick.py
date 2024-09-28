#harry potter clock trick
import cv2 as cv
import numpy as np
import time

video_capture=cv.VideoCapture(0)

fourcc=cv.VideoWriter_fourcc('m','p','4','v')
frame_width=int(video_capture.get(3))
frame_height=int(video_capture.get(4))
out=cv.VideoWriter('Demo.mp4',fourcc,10,(frame_width,frame_height),True)
time.sleep(3)
#splitting the background
bg=0
for i in range(30):
    result,bg=video_capture.read()
bg=np.flip(bg,axis=1)
#video analysing
while(video_capture.isOpened()):
    result,image=video_capture.read()
    if not result:
        break
    image=np.flip(image,axis=1)
    hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
    #create masks with coordinates to detect the color
    lower_blue=np.array([25,52,72])
    upper_blue=np.array([102,255,255])
    mask_all=cv.inRange(hsv,lower_blue,upper_blue)
    
    mask_all=cv.morphologyEx(mask_all,cv.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask_all=cv.morphologyEx(mask_all,cv.MORPH_DILATE,np.ones((3,3),np.uint8))

    mask2=cv.bitwise_not(mask_all)
    streamA=cv.bitwise_and(image,image,mask=mask2)
    streamB=cv.bitwise_and(bg,bg,mask=mask_all)

    #video in harddisk
    output=cv.addWeighted(streamA,1,streamB,1,0)
    out.write(output)
    cv.imshow("cloack_trick",output)
    if(cv.waitKey(25)==13):
        break
video_capture.release()
out.release()
cv.destroyAllWindows()
