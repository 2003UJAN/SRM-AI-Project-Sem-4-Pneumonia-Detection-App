from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("trained.h5")

# Define allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)[0][0]

        os.remove(file_path)  # Clean up

        result = "Pneumonia Detected" if prediction > 0.5 else "No Pneumonia"
        confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

        return jsonify({"result": result, "confidence": round(confidence * 100, 2)})

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    app.run(debug=True)
