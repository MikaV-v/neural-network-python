# добавим необходимый пакет с opencv
import cv2
# загружаем изображение и отображаем его
image = cv2.imread("face.jpg")
cv2.imshow("Original image", image)
cv2.waitKey(0)