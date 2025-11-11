import kagglehub
import tensorflow as tf
from tensorflow import keras

# Download latest version
path = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")

print("Path to dataset files:", path)


# Assuming your images are 28x28 pixels, if not, change the input shape accordingly
# You'll need to determine the actual dimensions of your images from the dataset
image_height = 28
image_width = 28
image_channels = 3 # For RGB images

model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same", input_shape=[image_height, image_width, image_channels]),
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
    keras.layers.Dense(10, activation="softmax")
])

model.summary()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Construct the path to an example image
# You might need to adjust the specific path based on the actual directory structure
# and the breed/image you want to display.
image_path = os.path.join(path, 'images', 'Images', 'n02085620-Chihuahua', 'n02085620_10074.jpg') # Example path

# Check if the file exists
if os.path.exists(image_path):
    # Load the image
    img = mpimg.imread(image_path)

    # Display the image
    plt.imshow(img)
    plt.axis('off') # Hide axis
    plt.show()

    # Print the shape and data type of the image array
    print("Image shape:", img.shape)
    print("Image data type:", img.dtype)

    # Split the image into R, G, and B channels
    r_channel = img[:, :, 0]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 2]

    print("\nShape of R channel:", r_channel.shape)
    print("values of first R channel", r_channel[0] / 255)
    print("Shape of G channel:", g_channel.shape)
    print("Shape of B channel:", b_channel.shape)

else:
    print(f"Image not found at: {image_path}")
    
import os
import xml.etree.ElementTree as ET

# Get the dataset path from the output of the first cell
dataset_path = path
annotation_path = os.path.join(dataset_path, 'annotations', 'Annotation')

# List to store annotation data
annotation_data = []
xml_files = []

# Walk through the directories and find XML files (they don't have .xml extension in this dataset)
for root_dir, _, files in os.walk(annotation_path):
    for file in files:
        # Skip hidden files and directories
        if not file.startswith('.'):
            xml_files.append(os.path.join(root_dir, file))

print(f"Found {len(xml_files)} XML annotation files.")

for file_path in xml_files:
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract image size
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        depth = int(size.find('depth').text)

        # Extract object information (assuming one object per annotation for now)
        obj = root.find('object')
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Store the extracted information
        annotation_data.append({
            'file_path': file_path,
            'breed': name,
            'width': width,
            'height': height,
            'depth': depth,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Now you have the annotation data in the 'annotation_data' list
# You can further process or analyze this data
print(f"Successfully processed {len(annotation_data)} annotation files.")