import joblib
import json
import numpy as np

# Load model components (deployed via /opt/ layer in Lambda)
model = joblib.load('/opt/rf_ids_model.joblib')
scaler = joblib.load('/opt/scaler.joblib')
label_encoder = joblib.load('/opt/label_encoder.joblib')

def lambda_handler(event, context):
    try:
        # Expect input like: { "features": [numeric_values_list] }
        features = event.get("features", [])
        if not isinstance(features, list) or not features:
            return {
                "statusCode": 400,
                "body": json.dumps("Invalid input format. Expecting key 'features' with a list of 78 values.")
            }

        # Reshape and scale
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)
        label = label_encoder.inverse_transform(prediction)[0]

        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": label})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error: {str(e)}")
        }
