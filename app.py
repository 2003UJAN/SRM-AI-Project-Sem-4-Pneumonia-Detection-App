from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import logging
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up uploads folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.chmod(UPLOAD_FOLDER, 0o755)  # Set proper permissions
    except Exception as e:
        logging.error(f"❌ Error creating uploads directory: {e}")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Allowed file types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Load trained model
try:
    model = tf.keras.models.load_model("trained.h5")
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    model = None

def preprocess_image(img_path):
    """Preprocesses the image for model prediction."""
    try:
        img = image.load_img(img_path, target_size=(150, 150))  # Resize to model input size
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        logging.error(f"❌ Error in image preprocessing: {e}")
        return None

@app.route("/predict", methods=["POST"])
def predict():
    """Handles image upload and model prediction."""
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs for details."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        if img_array is None:
            os.remove(file_path)
            return jsonify({"error": "Invalid image format"}), 400

        try:
            prediction = model.predict(img_array)[0][0]
        except Exception as e:
            os.remove(file_path)
            logging.error(f"❌ Prediction Error: {e}")
            return jsonify({"error": "Error making prediction"}), 500

        os.remove(file_path)  # Cleanup

        result = "Pneumonia Detected" if prediction > 0.5 else "No Pneumonia"
        confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)

        return jsonify({"result": result, "confidence": round(confidence * 100, 2)})

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True, threaded=True)
