import numpy as np
from flask import Flask, request, render_template
import pickle

# Create Flask app
flask_app = Flask(__name__)

# Load the trained model (Ensure model.pkl is in the correct path)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Ensure the form inputs are float values
    try:
        
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        
        
        prediction = model.predict(features)
        
    
        return render_template("index.html", prediction_text=f"The Predicted Crop is {prediction[0]}")
    except Exception as e:
        # Handle any errors that occur
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    flask_app.run(debug=True)
