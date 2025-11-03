import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

model = keras.applications.resnet50.ResNet50(weights="imagenet")

@app.route('/predict', methods=['GET'])
def predict():
    request_id = request.args.get('id', type=int)
    
    if request_id is None:
        return jsonify({"error": "Request ID is required"}), 400
    
    try:
        img_path = "./content/img.jpeg"
        img = Image.open(img_path)
        
        img_resized = tf.image.resize(img, [224, 224])
        
        inputs = keras.applications.resnet50.preprocess_input(img_resized[np.newaxis, ...])
        Y_proba = model.predict(inputs)
        
        top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=5)
        
        result = top_K[0][0][1]
        
        return jsonify({
            "request_id": request_id,
            "result": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)