import kagglehub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt

mixed_precision.set_global_policy('mixed_float16')

print("Downloading dataset... ")
path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")
dataset_path = pathlib.Path(path) / "images" / "Images"

image_height = 120
image_width = 120
batch_size = 16

print("Creating Data Pipeline... ")
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path, validation_split=0.2, subset="training", seed=123,
    image_size=(image_height, image_width), batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path, validation_split=0.2, subset="validation", seed=123,
    image_size=(image_height, image_width), batch_size=batch_size
)

model_path = "dog_spotter_model.keras"

if os.path.exists(model_path):
    print("Loading saved model...")
    model = keras.models.load_model(model_path)
else:
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(200).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 120

    model = keras.models.Sequential([
        keras.layers.Rescaling(1./255, input_shape=(image_height, image_width, 3)),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"), # Reduced filters
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(2),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax", dtype='float32')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'],jit_compile=True)
    print("Starting Training...")
    history =model.fit(train_ds,validation_data=val_ds, epochs=150)
    model.save(model_path)
    print(f"Model saved to {model_path}")

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
# --- Make a Prediction ---

# 1. Load a test image
# You can change this to any image path
test_image_path = f"{dataset_path}/n02085620-Chihuahua/n02085620_1007.jpg"
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