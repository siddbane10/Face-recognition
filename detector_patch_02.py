import cv2,os,sqlite3,pickle
import numpy as np
from PIL import Image 
flag=False
conn = sqlite3.connect('nbr_values.sqlite3')
cur=conn.cursor()
dir="/home/sid/Documents/Face-Recognition-master/"
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load(dir+'/trainer/trainer1.yml')
cascadePath = dir+"/Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = dir +'data-dump'
nbr_show=' '
nbr_status=' Not Recognised'
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX 
cur.execute('SELECT number FROM Nbr')
while True:
    ret, im =cam.read()
    
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for i in faces:
        x=i[0]
        y=i[1]
        w=i[2]
        h=i[3]
        nbr_predicted =recognizer.predict(gray[y:y+h,x:x+w])
        for row in cur:
            if nbr_predicted in row:
                nbr_status=' Recognised'
                nbr_show=str(nbr_predicted)
                flag=True
        
        if flag==False:
            nbr_status=' Not Recognised'
      

        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        
    cv2.putText(im, nbr_show + nbr_status, (230,50), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow('im',im)
    cv2.waitKey(10)    

cur.close()






