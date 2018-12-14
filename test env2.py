# https://habr.com/post/321834/
# tensorflow-gpu - 11.0
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json, Model
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import cv2, os
import time
from scipy.misc import toimage
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def ans(preds):
    print(preds)
    if preds > 0.26:
        return "Woman"
    else:
        return "Man"

sess = tf.InteractiveSession()
datagen = ImageDataGenerator(rescale=1. / 255)
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("mnist_model.h5")
model.compile(loss="binary_crossentropy", optimizer="SGD", metrics=["accuracy"])
activation_model = Model(inputs=model.input, outputs=model.layers[6].output)
buffre_dir = './buffer'
who_is_it = 'ME'
def nothing(x):
    pass
rect = np.array([0,0,0,0])
defa = np.array([0,0,0,0])
ans_list=[]
status = 1

cap = cv2.VideoCapture(0)
crop = cap

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cv2.namedWindow("Frame")
cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)


most_common=0
def reset(event):
    global ans_list, status, most_common
    status = 0
    ans_set = set(ans_list)
    most_common = None
    qty_most_common = 0
    for item in ans_set:
        qty = ans_list.count(item)
        if qty > qty_most_common:
            qty_most_common = qty
            most_common = item
    text.insert(1.0, most_common)
    ans_list = []
    status = 0
    text.get('1.0', 'end')
    text.tag_add('title', 1.0, '1.end')
    text.tag_config('title', font=("Verdana", 60, 'bold'), justify=CENTER)
    text.pack()

def start(event):
    global status
    text.delete(1.0, END)
    cap.release()
    status = 1


root = Tk()

text = Text(width=50, height=40)

Button_status_reset = Button(root, text = 'Show Answer', width=40, height=30)
Button_status_start = Button(root, text = 'start', width=40, height=30)

Button_status_start.bind('<Button-1>', start)
Button_status_start.bind('<Return>', start)


Button_status_reset.bind('<Button-1>', reset)
Button_status_reset.bind('<Return>', reset)

Button_status_start.pack()
Button_status_reset.pack()
text.pack()

while True:
    list_rect = rect.tolist()
    rectizm = rect
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    neighbours = cv2.getTrackbarPos("Neighbours", "Frame")

    faces = face_cascade.detectMultiScale(gray, 1.3, neighbours)
    for rect in faces:

        (x, y, w, h) = rect
        #cv2.imshow("", frame[y: y + h, x: x + w])
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('promeszh.jpg', frame[y: y + h, x: x + w])
        img = image.load_img('promeszh.jpg', target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.
        activation = activation_model.predict(x)
        preds = model.predict(x)
        #print(ans(preds))
        ans_list.append(ans(preds))
        if len(ans_list) == 40:
            ans_set = set(ans_list)
            most_common = None
            qty_most_common = 0
            for item in ans_set:
                qty = ans_list.count(item)
                if qty > qty_most_common:
                    qty_most_common = qty
                    most_common = item

        #for_neero_site.append(frame[y: y + h, x: x + w])
        os.remove("promeszh.jpg")


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if rect is rectizm:
        rect = np.array([0, 0, 0, 0])

    if key == 27:
        break
    #print(rect.tolist())
    if len(ans_list)==40:
        print("already answer:",most_common)
        print(ans_list)
        most_common = []
        ans_list = []
        print('wati')
        time.sleep(1)
        print('wati.')
        time.sleep(1)
        print('wati..')
        time.sleep(1)
        print('wati...')
        time.sleep(1)
        print('wati....')
        time.sleep(1)
        print('wati.....')
        time.sleep(1)
        print('wati......')
        time.sleep(1)


cap.release()
cv2.destroyAllWindows()