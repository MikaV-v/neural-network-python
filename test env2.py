# https://habr.com/post/321834/
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import cv2, os

root = Tk()
Button_status = Button(root, text = '4').pack(side = 'bottom')

sess = tf.InteractiveSession()
datagen = ImageDataGenerator(rescale=1. / 255)
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("mnist_model.h5")
model.compile(loss="binary_crossentropy", optimizer="SGD", metrics=["accuracy"])


buffre_dir = './buffer'
who_is_it = 'ME'
def nothing(x):
    pass
rect = np.array([0,0,0,0])
defa = np.array([0,0,0,0])
raspoznai = np.array([0,0,0,0])

cap = cv2.VideoCapture(0)
crop = cap

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cv2.namedWindow("Frame")
cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)

nan = 0
while True:

    if nan < 7:
        list_rect = rect.tolist()
        rectizm = rect
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        neighbours = cv2.getTrackbarPos("Neighbours", "Frame")

        faces = face_cascade.detectMultiScale(gray, 1.3, neighbours)
        for rect in faces:
            (x, y, w, h) = rect
            cv2.imshow("", frame[y: y + h, x: x + w])
            #for_neero_site.append(frame[y: y + h, x: x + w])
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if Button_status is True:
                cv2.imwrite(os.path.join('./buffer', str(nan) + '.jpg'), image[y: y + h, x: x + w])

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(20)
        if rect is rectizm:
            rect = np.array([0, 0, 0, 0])

        if key == 27:
            break
        print(rect.tolist())
    nan += 1


image_paths = './buffer'

img = image.load_img(2, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
def ans(preds):
    if preds > 0.5:
        return "Woman"
    else:
        return "Man"


root.mainloop()
cap.release()
cv2.destroyAllWindows()