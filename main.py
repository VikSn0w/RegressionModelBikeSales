import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error

# -------------------- Load and Clean Data --------------------
df = pd.read_csv("europe-motorbikes-zenrows.csv")
df = df.dropna()
df = df.drop(columns=["fuel", "gear", "offer_type", "link", "version"])

# Extract month and year from date
df["month"] = pd.to_datetime(df["date"], format="%m/%Y").dt.month
df["year"] = pd.to_datetime(df["date"], format="%m/%Y").dt.year
df = df.drop(columns=["date"])

# Filter out extreme prices
df = df[(df["price"] > 500) & (df["price"] < 50000)]

# Apply log transformation to the target variable (price)
y = np.log(df["price"] / 1000.0)  # Log transformation (price in thousands)
X = df.drop(columns=["price"])

# Encode categorical features
le_model = LabelEncoder()
le_version = LabelEncoder()
X["make_model"] = le_model.fit_transform(X["make_model"])

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------- RandomizedSearchCV for Hyperparameter Tuning --------------------
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

print(f"Best Hyperparameters: {random_search.best_params_}")

# -------------------- Evaluate the Best Model --------------------
best_rf_model = random_search.best_estimator_

# Cross-validation MSE
cv_scores = cross_val_score(best_rf_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE (in thousands of €): {-cv_scores.mean():.4f}")

# Evaluate on Test Set
y_pred = best_rf_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)

joblib.dump(best_rf_model, 'motorbike_price_model.joblib')
joblib.dump(scaler, 'scaler.joblib')  # Save the scaler for future use
print(f"Test MSE (in thousands of €): {test_mse:.4f}")


def predict_price(model, scaler, input_data):
    """
    Predict the price of a motorbike given input features.

    Parameters:
    - model: The trained model (e.g., Voting Regressor).
    - scaler: The scaler used to normalize the features during training.
    - input_data: A list or numpy array with the features of the motorbike.

    Returns:
    - The predicted price in thousands of euros.
    """
    # Ensure the input data is in a proper format (array-like, 2D)
    input_data_scaled = scaler.transform([input_data])[0]

    # Predict the price
    predicted_price = model.predict([input_data_scaled])[0]

    return predicted_price

# New input (a motorbike's details)
new_input = {
    "mileage": 15000,  # mileage (numeric)
    "power": 75,  # power (numeric)
    "make_model": "Ducati Monster 900",  # make_model (categorical)
    "month": 6,  # You must manually provide the correct month (from date)
    "year": 1999  # You must manually provide the correct year (from date)
}

# Encode the categorical variables using the same LabelEncoders
new_input_encoded = new_input.copy()
new_input_encoded["make_model"] = le_model.transform([new_input["make_model"]])[0]

# Create the input data array (including all features except price)
input_data = [
    new_input_encoded["mileage"],
    new_input_encoded["power"],
    new_input_encoded["make_model"],
    new_input_encoded["month"],
    new_input_encoded["year"]
]

print(input_data)

# Predict the price
predicted_price = predict_price(best_rf_model, scaler, input_data)

print(f"Predicted Price: €{np.exp(predicted_price) * 1000:.2f}")  # Exponentiate to get original price
