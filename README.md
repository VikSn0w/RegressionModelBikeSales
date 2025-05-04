# Motorbike Price Prediction Using Random Forest
## Description:
This notebook demonstrates how to build a machine learning model to predict the price of used motorbikes based on various features like mileage, power, make/model, version, and other factors. The model uses a Random Forest Regressor, and we employ several machine learning techniques such as data pre-processing, feature encoding, normalization, and hyperparameter tuning to achieve an accurate prediction.

The dataset has been provided by Mexwell on Keggle: https://www.kaggle.com/datasets/mexwell/motorbike-marketplace

### Model accuracy at the moment
 - Cross-validated MSE (in thousands of €): 0.0823
 - Test MSE (in thousands of €): 0.0780


## Steps:
1. Data Loading & Cleaning:
2. The motorbike dataset is loaded from a CSV file.
3. The dataset is cleaned by dropping missing values and irrelevant columns.
4. Additional features such as month and year are extracted from the date column.

## Feature Engineering:

We apply log transformation to the target variable (price) for better model performance.

Label Encoding is used to handle categorical features (make_model, version).

## Data Scaling:

Features are scaled using StandardScaler to normalize the input data and improve the performance of machine learning models.

## Model Training:

The dataset is split into training and test sets.

RandomizedSearchCV is used to perform hyperparameter tuning on a Random Forest model, optimizing its performance.

## Model Evaluation:

Cross-validation is used to evaluate the model on multiple splits of the data.

The model is evaluated using Mean Squared Error (MSE) on the test data.

Price Prediction:

A function is defined to predict the price of a new motorbike based on user input, and the price is returned in its original scale (non-log-transformed).

## Objective:
The primary goal of this notebook is to demonstrate the end-to-end process of training a machine learning model to predict the price of a motorbike, and to provide a way for users to predict the price of new motorbikes based on their features.

## Key Libraries Used:
- pandas: For data manipulation and cleaning.
- numpy: For numerical operations.
- sklearn: For machine learning models and preprocessing tools.
- joblib: For saving and loading the trained model and scaler.

## Example Input for Price Prediction:
- Mileage: 150 km
- Power: 218.0 hp
- Make/Model: Honda
- Version: CBR1000RR-R Fireblade SP
- Month: 3 (March)
- Year: 2020

The model predicts the motorbike's price based on these inputs, giving you an estimated value in euros.
