from PIL import Image, ImageDraw
import numpy
import cv2
# https://habr.com/post/163663/
# https://azure.microsoft.com/ru-ru/services/cognitive-services/face/
# https://habr.com/post/301096/
# https://proglib.io/p/50-python-projects/#log-in

mode = int(input('mode:'))  # Считываем номер преобразования.
image = Image.open("face.jpg")  # Открываем изображение.
# Сделац открывание фото в цикле переменная со списка (скан репозитория)


