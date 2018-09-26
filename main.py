from PIL import Image, ImageDraw
# https://habr.com/post/163663/
mode = int(input('mode:')) #Считываем номер преобразования.
image = Image.open("face.jpg") #Открываем изображение.
#Сделац открывание фото в цикле