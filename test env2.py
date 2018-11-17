import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
import pandas as pd
import h5py
import matplotlib.pyplot as plt

def nothing(x):
    pass
rect = np.array([0,0,0,0])
defa = np.array([0,0,0,0])

cap = cv2.VideoCapture(0)
crop = cap

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cv2.namedWindow("Frame")
cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)

while True:
    list_rect = rect.tolist()
    rectizm = rect
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    neighbours = cv2.getTrackbarPos("Neighbours", "Frame")

    faces = face_cascade.detectMultiScale(gray, 1.3, neighbours)
    for rect in faces:
        (x, y, w, h) = rect
        cv2.imshow("", frame[y: y + h, x: x + w])
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(80)
    if rect is rectizm:
        rect = np.array([0, 0, 0, 0])

    if key == 27:
        break
    print(rect.tolist())

cap.release()
cv2.destroyAllWindows()