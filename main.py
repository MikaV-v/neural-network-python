
# https://habr.com/post/318846/
# https://habr.com/post/163663/
# https://azure.microsoft.com/ru-ru/services/cognitive-services/face/
# https://habr.com/post/301096/

# Импортируем необходимые модули
import cv2, os
import numpy as np
from PIL import Image

# Для детектирования лиц используем каскады Хаара
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Для распознавания используем локальные бинарные шаблоны
recognizer = cv2.createLBPHFaceRecognizer(1,8,8,8,123)


#mode = int(input('mode:'))  # Считываем номер преобразования.
#image = Image.open("face.jpg")  # Открываем изображение.
# Сделац открывание фото в цикле переменная со списка (скан репозитория)
