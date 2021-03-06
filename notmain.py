from PIL import Image, ImageDraw
import numpy as np
import cv2, os

nan = 1
# https://habr.com/post/163663/
# https://azure.microsoft.com/ru-ru/services/cognitive-services/face/
# https://habr.com/post/301096/

cascadePath = "haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath)

# recognizer = cv2.face.LBPHFaceRecognizer_create(1,8,8,8,123)

# Для распознавания используем локальные бинарные шаблоны
# Открываем изображение.
# Сделац открывание фото в цикле переменная со списка (скан репозитория)


def get_images(path):
    # Ищем все фотографии и записываем их в image_paths
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.happy')]

    images = []
    labels = []

    for image_path in image_paths:
        # Переводим изображение в черно-белый формат и приводим его к формату массива
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        #rotated = image.rotate(90)
        # Из каждого имени файла извлекаем номер человека, изображенного на фото
        #subject_number = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))

        # Определяем области где есть лица
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.01, minNeighbors=1, minSize=(110, 110))
        # Если лицо нашлось добавляем его в список images, а соответствующий ему номер в список labels
        for (x, y, w, h) in faces:
            global nan
            nan += 1
            images.append(image[y: y + h, x: x + w])
            #labels.append(subject_number)
            # В окне показываем изображение
            #cv2.imshow("", image[y: y + h, x: x + w])

            cv2.imwrite(os.path.join('./buffer', str(nan)+'.jpg'), image[y: y + h, x: x + w])
            cv2.waitKey(1)
    return images, labels


path = './buffer'
# Получаем лица и соответствующие им номера
images, labels = get_images(path)
cv2.destroyAllWindows()

# Обучаем программу распознавать лица
#recognizer.train(images, np.array(labels))
#print(recognizer.train(images, np.array(labels)))