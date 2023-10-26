import cv2
import numpy as np 
from PIL import Image
import os

path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples= []
    ids = []

    for imagePath in image_paths:

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            face_samples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return face_samples, ids
print('\n [INFO] Training data...')
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
recognizer.write( 'trainer/trainer.yml')
print('done')