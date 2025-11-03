import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt

model = keras.applications.resnet50.ResNet50(weights="imagenet")

img_path = "img.jpeg"
img = Image.open(img_path)

img_resized = tf.image.resize(img, [224, 224])

inputs = keras.applications.resnet50.preprocess_input(img_resized[np.newaxis, ...])
Y_proba = model.predict(inputs)
print(Y_proba.shape)

top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=5)

for class_id, name, y_proba in top_K[0]:
  print("  {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100)) 