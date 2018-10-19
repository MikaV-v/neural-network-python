
# https://habr.com/post/318846/
# https://habr.com/post/163663/
# https://azure.microsoft.com/ru-ru/services/cognitive-services/face/
# https://habr.com/post/301096/




"""
уже НЕ САМОЕ ГЛАВНОЕ ДОУСТАНОВИТЬ БИБЛИОТЕКУ OPENCV2
"""


# https://habr.com/post/321834/ вот ствол

# Импортируем необходимые модули
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential, Model
# from keras.applications.inception_v3 import InceptionV3
# from keras.callbacks import ModelCheckpoint
# from keras.optimizers import SGD
#
# from keras import backend as K
# K.set_image_dim_ordering('th')
#
# import numpy as np
# import pandas as pd
# import h5py
#
# import matplotlib.pyplot as plt
#
#
#
# inc_model=InceptionV3(include_top=False,weights='imagenet',input_shape=((3, 150, 150)))
# Сделац открывание фото в цикле переменная со списка (скан репозитория)


import opencv as cv2
import numpy as np
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.createLBPHFaceRecognizer(1,8,8,8,123)