import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import logging

logging.basicConfig(level=logging.DEBUG)

# Create Flask app
flask_app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open("model.pkl", "rb"))
    logging.debug("Model loaded successfully")
except FileNotFoundError:
    model = None
    logging.error("Error: model.pkl not found!")

# Flask Routes
@flask_app.route("/")
def home():
    logging.debug("Flask Home route accessed")
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    if not model:
        logging.error("Model not loaded, returning error response")
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        float_features = [float(x) for x in request.form.values()]
        features = np.array([float_features])
        
        # Make prediction
        prediction = model.predict(features)
        logging.debug(f"Prediction made: {prediction[0]}")
        
        return jsonify({"prediction": f"The Predicted Crop is {prediction[0]}"})
    
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    logging.debug("Flask app is running on port 5000...")
    flask_app.run(debug=True)
