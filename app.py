from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

model = load_model('confiable.keras')
possible_labels = ['0_normal', '1_ulcerative_colitis', '2_polyps', '3_esophagitis']

def preprocess_image(image):
    image = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    image = cv2.resize(image, (320, 320))
    image = np.expand_dims(image, axis=0)
    
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    image = preprocess_image(file.read())
    prediction = model.predict(image)
    label = possible_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return jsonify({'label': label, 'confidence': confidence})

if __name__ == '__main__':
    # Iniciar la aplicación Flask en el puerto 5000
    app.run(host='0.0.0.0', port=5000)
