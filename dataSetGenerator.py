import cv2
import numpy as np
cam = cv2.VideoCapture(0)

dir="/home/sid/Face-Recognition-master"
detector=cv2.CascadeClassifier(dir+'/Classifiers/face.xml')
i=0
offset=50
#name=raw_input('enter your id')
name=raw_input("Enter your name ")

while True:
    
    ret, im = cam.read()
    
    #print im
    #im = cv2.cv.QueryFrame(cam)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow('input',im)
    cv2.waitKey(0)
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=8, minSize=(100,100), flags=cv2.CASCADE_SCALE_IMAGE)
    print faces
    for j in faces:
        x=j[0]
        y=j[1]
        w=j[2]
        h=j[3]
        
        i=i+1
        cv2.imwrite(dir+"/dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        cv2.imshow('im',im[y-offset:y+h+offset,x-offset:x+w+offset])
        print dir+"/dataSet/face-"+name +'.'+ str(i) + ".jpg"
        im3=cv2.imread(dir+"/dataSet/face-"+name +'.'+ str(i) + ".jpg")
        cv2.imshow('hello',im3)
        cv2.waitKey(200)
    if i>19:
        cam.release()
        cv2.destroyAllWindows()
        break

