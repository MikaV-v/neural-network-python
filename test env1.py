from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.optimizers import Adam


# Каталог с данными для обучения
train_dir = 'train'
# Каталог с данными для проверки
val_dir = 'val'
# Каталог с данными для тестирования
test_dir = 'test'
# Размеры изображения
img_width, img_height = 150, 150
# Размерность тензора на основе изображения для входных данных в нейронную сеть
# backend Tensorflow, channels_last
input_shape = (img_width, img_height, 3)
# Размер мини-выборки
batch_size = 10
nb_train_samples = 1260
nb_validation_samples = 270
nb_test_samples = 270

# Загружаем предварительно обученную нейронную сеть VGG16
vgg16_net = VGG16(weights='imagenet', include_top=False,
                  input_shape=(150, 150, 3))

# "Замораживаем" веса предварительно обученной нейронной сети VGG16
vgg16_net.trainable = True

# Создаем составную нейронную сеть на основе VGG16
# Создаем последовательную модель Keras
model = Sequential()
# Добавляем в модель сеть VGG16 вместо слоя
model.add(vgg16_net)
# Добавляем в модель новый классификатор



model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(1))
model.add(Activation('sigmoid'))





# Компилируем составную нейронную сеть
model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])

# Создаем генератор изображений на основе класса ImageDataGenerator.
# Генератор делит значения всех пикселов изображения на 255.
datagen = ImageDataGenerator(rescale=1. / 255)

# Генератор данных для обучения на основе изображений из каталога
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Генератор данных для проверки на основе изображений из каталога
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Генератор данных для тестирования на основе изображений из каталога
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Обучаем модель с использованием генераторов
def madell():
    model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=7,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)

# Генерируем описание модели в формате json
model_json = model.to_json()
# Записываем модель в файл
json_file = open("mnist_model.json", "w")
json_file.write(model_json)
json_file.close()

madell()
# Оцениваем качество работы сети с помощью генератора
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.3f%%" % (scores[1]*100))
model.save_weights("mnist_model.h5")