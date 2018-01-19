import cv2
import os
import numpy as np
from PIL import Image 
dir="/home/sid/Face-Recognition-master"
recognizer = cv2.face.createLBPHFaceRecognizer()
cascadePath = dir+"/Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = dir+'/dataSet'

def get_images_and_labels(path):
     image_paths = [os.path.join(path, f) for f in os.listdir(path)]
     
     # images will contains face images
     images = []
     # labels will contains the label that is assigned to the image
     labels = []
     for image_path in image_paths:
         print image_path
         # Read the image and convert to grayscale
         image_pil = Image.open(image_path).convert('L')
         # Convert the image format into numpy array
         image = np.array(image_pil, 'uint8')
         # Get the label of the image
         nbr = os.path.split(image_path)[1].split(".")[0].replace("face-", "")
         print nbr
         print "someval1"
         nbr=int(''.join(str(ord(c)) for c in nbr))
         print nbr
         print "sval2"
         # Detect the face in the image
         faces = faceCascade.detectMultiScale(image)
         # If face is detected, append the face to images and the label to labels
         for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             labels.append(nbr)
             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             cv2.waitKey(10)
     # return the images list and labels list
     return images, labels


images, labels = get_images_and_labels(path)
cv2.imshow('test',images[0])
cv2.waitKey(1)

recognizer.train(images, np.array(labels))
recognizer.save(dir+'/trainer/trainer1.yml')
cv2.destroyAllWindows()
