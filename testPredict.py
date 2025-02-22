import sys
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Get features from command-line arguments
features = list(map(float, sys.argv[1:]))

# Handle missing savings (if applicable)
if len(features) < 7:
    features.append(0)  # Default savings to 0 if missing

# Apply logarithmic scaling to num_customer_reviews and num_seller_reviews
# Handle zero values by replacing them with 1 (or a small positive value)
features[1] = np.log(features[1]) if features[1] > 0 else np.log(1)  # Transform num_customer_reviews
features[3] = np.log(features[3]) if features[3] > 0 else np.log(1)  # Transform num_seller_reviews

# Scale the features
features_scaled = scaler.transform([features])

# Predict the personalized rating
prediction = model.predict(features_scaled)
print(round(prediction[0], 2))  # Round to 2 decimal places