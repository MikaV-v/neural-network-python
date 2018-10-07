
# https://habr.com/post/318846/
# https://habr.com/post/163663/
# https://azure.microsoft.com/ru-ru/services/cognitive-services/face/
# https://habr.com/post/301096/




"""
САМОЕ ГЛАВНОЕ ДОУСТАНОВИТЬ БИБЛИОТЕКУ OPENCV2
"""




# Импортируем необходимые модули
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt



inc_model=InceptionV3(include_top=False,
                      weights='imagenet',
                      input_shape=((3, 150, 150)))
# Сделац открывание фото в цикле переменная со списка (скан репозитория)
