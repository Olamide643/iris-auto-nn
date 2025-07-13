from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

# Load model, scaler, and class names
model = tf.keras.models.load_model("./iris_nn_model.h5", compile = False)
scaler = joblib.load("./iris_scaler.pkl")
class_names = joblib.load("./class_names.pkl")

print("Yes")
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Neural Network Iris API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_df = pd.DataFrame([data])
        input_scaled = scaler.transform(input_df)

        prediction_probs = model.predict(input_scaled)
        prediction = np.argmax(prediction_probs[0])
        class_name = class_names[prediction]

        return jsonify({
            "prediction": int(prediction),
            "class_name": class_name,
            "probabilities": prediction_probs[0].tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
