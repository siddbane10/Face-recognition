import cv2,os
import numpy as np
from PIL import Image 
import pickle
dir="/home/sid/Face-Recognition-master"
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load(dir+'/trainer/trainer1.yml')
cascadePath = dir+"/Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = dir +'dataSet'

#cam = cv2.VideoCapture(0)
im = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX #Creates a font
while True:
    #ret, im =cam.read()
  
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for i in faces:
        x=i[0]
        y=i[1]
        w=i[2]
        h=i[3]
        nbr_predicted = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        print (nbr_predicted)
        if(nbr_predicted==1237832213):
             nbr_predicted='Shivam'
        elif(nbr_predicted==837368):
             nbr_predicted='Siddhant'
        elif(nbr_predicted==11697110118105):
            nbr_predicted='tanvi'
        elif(nbr_predicted==9711010697108105):
            nbr_predicted='anjali'
        else: nbr_predicted='Unone'

        #cv2.cv.PutText(cv2.cv.fromarray(im),str(nbr_predicted)+"--"+str(conf), (x,y+h),font, 255) #Draw the text
        
        cv2.putText(im, nbr_predicted, (230,50), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
        
        cv2.imshow('im',im)
        cv2.waitKey(10)









