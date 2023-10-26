import cv2 
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
face_id = input('\n Id of the face <return> ==>  ')

print("\n [INFORMATION] Create Camera...")
count = 0
while True: 
    _, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
    for (x,y,w,h) in faces:  #x la hoanh do(cot), y la tung do(dong), w va h la chieu dai, chieu rong
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        count += 1

        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", img_gray[y:y+h, x:x+w])
        cv2.imshow('image',img)
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break

print("\n [INFORMATION] Exit")
cap.release()
