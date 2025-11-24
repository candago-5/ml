"""Reconstruct model architecture, load weights from `dog_spotter_model.keras`, and save a fixed .keras file.
Run from: /home/caiquefrd/fatec/dogFinder/ml/source/tests/examples/mobileNetV2
Inside the example venv: `source ../venv/bin/activate` then `python3 save_fixed_model.py`
"""
import zipfile
import pathlib
import traceback
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

try:
    import tensorflow as tf
    p = pathlib.Path('dog_spotter_model.keras')
    if not p.exists():
        raise FileNotFoundError('dog_spotter_model.keras not found in current dir')

    # Extract internal weights HDF5
    with zipfile.ZipFile(p, 'r') as z:
        names = z.namelist()
        candidate = None
        for n in names:
            if n.endswith('.h5') or 'weights' in n:
                candidate = n
                break
        if candidate is None:
            raise RuntimeError(f'No internal HDF5 weights found in archive; contents: {names}')
        extract_path = pathlib.Path('_extracted_model_weights.h5')
        with z.open(candidate) as src, open(extract_path, 'wb') as dst:
            dst.write(src.read())
    print(f'Extracted weights to {extract_path}')

    # Reconstruct the training architecture (same as mobileNetV2.py)
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
        tf.keras.layers.Input(shape=(image_height, image_width, 3)),
        tf.keras.layers.Rescaling(1./127.5, offset=-1),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])

    # Load weights by name first (more tolerant)
    try:
        model.load_weights(str(extract_path), by_name=True)
        print('Loaded weights by name into reconstructed model')
    except Exception:
        print('Failed to load weights by name; trying strict load')
        traceback.print_exc()
        model.load_weights(str(extract_path))
        print('Loaded weights (strict) into reconstructed model')

    # Save fixed model archive
    out_path = pathlib.Path('dog_spotter_model_fixed.keras')
    model.save(out_path)
    print(f'Saved reconstructed model to: {out_path}')

except Exception:
    print('Error while saving fixed model:')
    traceback.print_exc()
