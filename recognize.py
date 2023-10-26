import cv2
import numpy as np
import os


recognize =  cv2.face.LBPHFaceRecognizer_create()
recognize.read('trainer/trainer.yml')
cascadepath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadepath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0 

names = ['Obama','Bao','Vy']
cam = cv2.VideoCapture(0)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)


while True:

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


    faces = faceCascade.detectMultiScale(gray, 
                                          scaleFactor=1.2,
                                          minNeighbors=5,
                                          minSize=(int(minW),int(minH)),
                                         )
    for (x,y,w,h) in faces:
        
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
        id, confidence = recognize.predict(gray[y:y+h, x:x+w])

        if (confidence <100):

            id = names[id]
            confidence = " {0}%".format(round(100-confidence))
        else:
            id = 'unknown'
            confidence =  " {0}%".format(round(100-confidence))
        cv2.putText(img, str(id), (x+5, y-5), font, 1 ,(255,255,255),2)
        cv2.putText(img, str(confidence), (x+80, y-5), font,1,(255,255,0),1)


    cv2.imshow('nhandienkhuonmat', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
    
print("\n [INFO] EXIT")
cam.release()
cv2.destroyAllWindows()