from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the model and preprocessing files
model = load_model("takeprofit_model_with_symbols.keras")
metadata = joblib.load("model_metadata.pkl")  # Scaler metadata
scaler = metadata["scaler"]
features = metadata["features"]
le = joblib.load("symbol_encoder.pkl")  # LabelEncoder for Symbol column

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
        input_rows["SymbolEncoded"] = le.transform(input_rows["Symbol"])

        # Normalize features and append Symbol as the last column
        scaled_features = scaler.transform(input_rows[features])
        X_numerical = np.array(scaled_features).reshape(1, len(input_rows), -1)
        X_symbol = np.array(input_rows["SymbolEncoded"].values[-1:]).reshape(1, 1)

        # Make a prediction and take the absolute value
        prediction = model.predict([X_numerical, X_symbol])
        absolute_takeprofit = abs(prediction[0][0])

        # Return the absolute value of the prediction
        return jsonify({"TakeProfit": absolute_takeprofit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)