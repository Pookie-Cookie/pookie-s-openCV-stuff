import cv2
import numpy as np
import matplotlib
import random

trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)

while True:
    success, img = cap.read()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_coords = trained_data.detectMultiScale(gray_image)
    print(face_coords)
    for (x, y, w, h) in face_coords:
        cv2.rectangle(img, (x, y), (x+w, y+h), (random.randrange(255), random.randrange(255), random.randrange(255)), 2)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
