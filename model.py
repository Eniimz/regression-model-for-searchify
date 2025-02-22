# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Step 1: Prepare the dataset
# Features: [customer_rating, log_num_customer_reviews, seller_rating, log_num_seller_reviews, savings, listed_price, current_price]
# Target: personalized_rating
X = [
    [4.5, np.log(100), 4.7, np.log(50), 50, 200, 150],  # Product 1
    [3.8, np.log(50), 4.2, np.nan, np.nan, 150, 120],  # Product 2 (missing savings and num_seller_reviews)
    [5.0, np.log(200), 4.9, np.log(30), 30, 250, 220],   # Product 3
    [4.2, np.log(80), 4.5, np.log(40), 20, 180, 160],    # Product 4
    [4.7, np.log(120), 4.8, np.log(60), np.nan, 220, 200],  # Product 5 (missing savings)
    # [4.9, np.log(300), 4.8, np.log(50000), 0, 300, 300],  # Similar to Case 5
    # [4.8, np.log(400), 4.7, np.log(10000), 10, 250, 240],  # High ratings and reviews
    # [4.7, np.log(200), 4.9, np.log(20000), 20, 200, 180], 
]

# Target: Personalized ratings
y = [4.7, 4.0, 4.8, 4.3, 4.6, 4.8, 4.6]

# Step 2: Handle missing data
# Impute missing values with the mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X_imputed = imputer.fit_transform(X)

print("Imputed Data:")
print(X_imputed)

# Step 3: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print("Scaled Data:")
print(X_scaled)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Step 7: Save the trained model and scaler to files
joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')