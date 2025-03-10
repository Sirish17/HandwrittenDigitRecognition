from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses most logs

app = Flask(__name__)

# Load the trained model
model_path = "models/mnist_cnn_model_tuned.keras"
model = tf.keras.models.load_model(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files["file"]
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Make sure the image has a black background and white digit (invert if necessary)
        if np.mean(image) > 127:
            image = cv2.bitwise_not(image)

        # Resize with correct interpolation
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize the image (convert to float32 and scale to [0,1])
        image = image.astype("float32") / 255.0
        
        # Ensure correct dimensions: (1, 28, 28, 1) - Batch, Height, Width, Channels
        image = np.expand_dims(image, axis=[0, -1])

        # Predict
        prediction = model.predict(image)
        digit = int(np.argmax(prediction))

        return jsonify({"digit": digit, "confidence": float(np.max(prediction))})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)
