# app.py

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the model and preprocessing files
model_path = os.path.join(os.path.dirname(__file__), "takeprofit_model_with_symbols.keras")
model = load_model(model_path)

metadata_path = os.path.join(os.path.dirname(__file__), "model_metadata.pkl")
metadata = joblib.load(metadata_path)
scaler = metadata.get("scaler")
features = metadata.get("features")

encoder_path = os.path.join(os.path.dirname(__file__), "symbol_encoder.pkl")
le = joblib.load(encoder_path)

if not scaler or not features:
    raise ValueError("Invalid or missing metadata keys in 'model_metadata.pkl'")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the TakeProfit Prediction API"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse the input JSON
        input_data = request.get_json()
        if "data" not in input_data:
            return jsonify({"error": "Invalid input format. 'data' key is required."}), 400

        # Convert the input data to a DataFrame
        input_rows = pd.DataFrame(input_data["data"], columns=features + ["Symbol"])

        # Convert numeric columns to float
        for col in features:
            input_rows[col] = input_rows[col].astype(float)

        # Encode Symbol column
        try:
            input_rows["SymbolEncoded"] = le.transform(input_rows["Symbol"])
        except ValueError as e:
            return jsonify({"error": f"Invalid symbol in input: {str(e)}"}), 400

        # Normalize features and append Symbol as the last column
        scaled_features = scaler.transform(input_rows[features])
        X_numerical = np.array(scaled_features).reshape(1, len(input_rows), -1)
        X_symbol = np.array(input_rows["SymbolEncoded"].values[-1:]).reshape(1, 1)

        # Make a prediction and take the absolute value
        prediction = model.predict([X_numerical, X_symbol])
        absolute_takeprofit = float(abs(prediction[0][0]))

        # Return the absolute value of the prediction
        return jsonify({"TakeProfit": absolute_takeprofit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)
