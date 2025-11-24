import kagglehub
import tensorflow as tf
from tensorflow import keras
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
import numpy as np


# Download latest version
path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")

print("Path to dataset files:", path)


dataset_path = "/home/caiquefrd/.cache/kagglehub/datasets/jessicali9530/stanford-dogs-dataset/versions/2/images/Images"
data_dir = pathlib.Path(dataset_path)

image_height = 240
image_width = 240
batch_size = 32
image_channels = 3


train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  
    subset="training",     
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",   
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size
)

print("Classes found:", train_ds.class_names)



model_path = "dog_spotter_model.keras"

if os.path.exists(model_path):
    print("Loading saved model...")
    model = keras.models.load_model(model_path)
else:
    print("No saved model found. Training a new model...")
    model = keras.models.Sequential([
        keras.layers.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
        keras.layers.Conv2D(64, 7, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(120, activation="softmax")
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=10)
    model.save(model_path)
    print(f"Model saved to {model_path}")


# --- Make a Prediction ---

# 1. Load a test image
# You can change this to any image path
test_image_path = f"{data_dir}/n02085620-Chihuahua/n02085620_1007.jpg"
img = tf.keras.utils.load_img(
    test_image_path, target_size=(image_height, image_width)
)

# 2. Preprocess the image
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

# 3. Make a prediction
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# 4. Decode and print the result
class_names = train_ds.class_names
predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print(f"This image most likely belongs to {predicted_class} with a {confidence:.2f} percent confidence.")