import kagglehub
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
import pathlib
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import os
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

model = keras.applications.resnet50.ResNet50(weights="imagenet")

@app.route('/predict', methods=['GET'])
def predict():
    request_id = request.args.get('id', type=int)
    
    # Change to the image id sent by user
    img_path = "./content/gante.jpeg"
    
    if request_id is None:
        return jsonify({"error": "Request ID is required"}), 400
    
    try:
        mixed_precision.set_global_policy('mixed_float16')

        path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")
        dataset_path = pathlib.Path(path) / "images" / "Images"

        image_height = 240
        image_width = 240
        batch_size = 32


        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path, validation_split=0.2, subset="training", seed=123,
            image_size=(image_height, image_width), batch_size=batch_size
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_path, validation_split=0.2, subset="validation", seed=123,
            image_size=(image_height, image_width), batch_size=batch_size
        )

        model_path = "dog_spotter_model.keras"

        # Try to load an existing model; if loading fails (corrupt or incompatible),
        # back it up and continue to train a fresh model.
        model = None
        if os.path.exists(model_path):
            print("Loading saved model...")
            try:
                model = keras.models.load_model(model_path)
            except Exception as e:
                print("Warning: failed to load saved model:", e)
                # backup the broken model file so training can proceed
                try:
                    backup_path = model_path + ".broken"
                    os.rename(model_path, backup_path)
                    print(f"Backed up broken model to: {backup_path}")
                except Exception as e2:
                    print("Also failed to back up the model file:", e2)

        if model is None:
            ## Train a new model
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.shuffle(200).prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

            num_classes = 120

            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(image_height, image_width, 3),
                include_top=False,
                weights='imagenet'
            )

            base_model.trainable = False

            model = keras.models.Sequential([
                keras.layers.Rescaling(1./127.5, offset=-1, input_shape=(image_height, image_width, 3)),

                base_model,

                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dropout(0.2),

                keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
            ])

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
                jit_compile=True
            )

            model.summary()
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=50
            )

            model.save(model_path)
            print(f"Model saved to {model_path}")



        def predict_dog_breed(image_path, model, class_names):
            if not os.path.exists(image_path):
                print(f"Error: Image {image_path} not found.")
                return

            img = load_img(image_path, target_size=(240, 240))
            img_array = img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_batch, verbose=0)

            n = 50
            top_n_indices = np.argsort(predictions[0])[::-1][:n]

            print(f"\n--- PREDICTION RESULTS FOR: {image_path} ---")

            for i in top_n_indices:
                breed = class_names[i]
                confidence = predictions[0][i] * 100
                print(f"{breed:<25} {confidence:.2f}%")
                
            return top_n_indices[0]

        # Get the classes based on dataset
        class_names = train_ds.class_names

        result = predict_dog_breed(img_path, model, class_names)
        
        #Output the result as JSON
        return jsonify({
            "request_id": request_id,
            "result": class_names[result]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)