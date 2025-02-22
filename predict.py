from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model and scaler
model = joblib.load('/home/eniim/Try/whtswrong/linear-regressoin/linear_regression_model.pkl')
scaler = joblib.load('/home/eniim/Try/whtswrong/linear-regressoin/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get product details from the request
    data = request.json

    # Prepare the input features
    customer_rating = min(data.get('customerRating', 0), 5)  # Clamp customer rating to max 5
    num_customer_reviews = data.get('numCustomerReviews', 0)
    seller_rating = min(data.get('sellerRating', 0), 5)  # Clamp seller rating to max 5
    num_seller_reviews = data.get('numSellerReviews', 0)
    savings = data.get('savings', 0)
    listed_price = data.get('listedPrice', 0)
    current_price = data.get('currentPrice', 0)

    # Apply logarithmic scaling to num_customer_reviews and num_seller_reviews
    num_customer_reviews_log = np.log(num_customer_reviews) if num_customer_reviews > 0 else np.log(1)
    num_seller_reviews_log = np.log(num_seller_reviews) if num_seller_reviews > 0 else np.log(1)

    # Prepare the feature array
    features = [
        customer_rating,
        num_customer_reviews_log,
        seller_rating,
        num_seller_reviews_log,
        savings,
        listed_price,
        current_price
    ]

    # Scale the features
    features_scaled = scaler.transform([features])

    # Predict the personalized rating
    prediction = model.predict(features_scaled)[0]

    # Ensure the predicted rating is also clamped to a maximum of 5
    prediction = min(prediction, 5)

    # Return the predicted rating as JSON
    return jsonify({'predictedRating': round(prediction, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Run the API on port 5001