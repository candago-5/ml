# save as load_check.py and run: python load_check.py
import traceback
import zipfile
import pathlib
import os
from tensorflow.keras import mixed_precision

# Use the same policy as training if mixed precision was used.
mixed_precision.set_global_policy('mixed_float16')

try:
    import tensorflow as tf
    # First try to load the full model (this may fail if formats differ)
    try:
        model = tf.keras.models.load_model('dog_spotter_model.keras', compile=False)
        print('Loaded full model OK')
    except Exception:
        print('Full `load_model` failed; attempting to extract weights and load into reconstructed model...')
        traceback.print_exc()

        # Locate the .keras archive and extract the internal HDF5 weights file
        archive_path = pathlib.Path('dog_spotter_model.keras')
        if not archive_path.exists():
            raise FileNotFoundError(f"Model archive not found at {archive_path}")

        with zipfile.ZipFile(archive_path, 'r') as z:
            names = z.namelist()
            # Look for the canonical weights file name inside a Keras .keras archive
            candidate = None
            for n in names:
                if n.endswith('.h5') or 'weights' in n:
                    candidate = n
                    break
            if candidate is None:
                raise RuntimeError(f"No internal weights HDF5 found in {archive_path}; contents: {names}")

            extract_path = pathlib.Path('_extracted_model_weights.h5')
            with z.open(candidate) as src, open(extract_path, 'wb') as dst:
                dst.write(src.read())
            print(f"Extracted internal weights file to: {extract_path}")

        # Reconstruct the architecture used when training and load weights by name
        image_height = 240
        image_width = 240
        num_classes = 120

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(image_height, image_width, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = tf.keras.models.Sequential([
            tf.keras.layers.Rescaling(1./127.5, offset=-1, input_shape=(image_height, image_width, 3)),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
        ])

        # Try loading weights. Use `by_name=True` to be more tolerant to small config differences.
        try:
            model.load_weights(str(extract_path), by_name=True)
            print('Weights loaded into reconstructed model (by_name=True).')
        except Exception:
            print('Failed to load weights by name; trying strict load...')
            traceback.print_exc()
            try:
                model.load_weights(str(extract_path))
                print('Weights loaded into reconstructed model (strict).')
            except Exception:
                print('Strict weights load also failed. Showing final traceback:')
                traceback.print_exc()

except Exception:
    print('Unexpected error during model inspection:')
    traceback.print_exc()