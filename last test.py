from keras.models import model_from_json
from keras.preprocessing import image
from keras_applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os
buffre_dir = 'buffer'

sess = tf.InteractiveSession()
datagen = ImageDataGenerator(rescale=1. / 255)
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("mnist_model.h5")
model.compile(loss="binary_crossentropy", optimizer="SGD", metrics=["accuracy"])
# buffre_gen = datagen.flow_from_directory(
#      buffre_dir,
#      target_size=(150, 150),
#      batch_size=10,
#      class_mode=None,
#      shuffle=False
#  )

img = image.load_img('3.jpg', target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

preds = model.predict(x)


def ans(preds):
    if preds>0.5:
        return "Woman"
    else:
        return "Man"
print(ans(preds))
print(preds)
