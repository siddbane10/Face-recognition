import cv2
import os
import numpy as np
from PIL import Image 
import shutil
import sqlite3

conn = sqlite3.connect('nbr_values.sqlite3')
cur = conn.cursor()


nbr_list=[]
dir="/home/sid/Documents/Face-Recognition-master"
recognizer = cv2.face.createLBPHFaceRecognizer()
cascadePath = dir+"/Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = dir+'/dataSet'
path2 = dir+'/data-dump'
def get_images_and_labels(path):
     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    
     
     images = []
    
     labels = []
     for image_path in image_paths:
         print image_path
         
         image_pil = Image.open(image_path).convert('L')
         
         image = np.array(image_pil, 'uint8')
        
         nbr = os.path.split(image_path)[1].split(".")[0].replace("face-", "")
         print nbr

         nbr=int(''.join(str(ord(c)) for c in nbr))

         print nbr
         
         
         if nbr in nbr_list:
            print " "
         else:
            nbr_list.append(nbr)
            cur.execute('INSERT INTO Nbr(number) VALUES (?)', (nbr,))
            conn.commit()
            
         
         shutil.copy2(image_path,path2)
         os.remove(image_path)
        
      
         faces = faceCascade.detectMultiScale(image)
        
         for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             labels.append(nbr)
             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             cv2.waitKey(10)
  
     return images, labels


images, labels = get_images_and_labels(path)

print nbr_list
recognizer.train(images, np.array(labels))
recognizer.save(dir+'/trainer/trainer1.yml')
cur.close()
cv2.destroyAllWindows()
