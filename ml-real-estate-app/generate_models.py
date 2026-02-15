import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
import os

# Paths
HOUSING_CSV = 'shared/models/housing_data.csv'
LOAN_CSV = 'shared/models/loan_data.csv'
MODEL_DIR = 'ml-real-estate-app/models'

os.makedirs(MODEL_DIR, exist_ok=True)

print("--- Training House Price Model ---")
try:
    df_house = pd.read_csv(HOUSING_CSV)
    
    # Mapping Condition
    condition_map = {'Bad': 1, 'Okay': 2, 'Fair': 3, 'Good': 4, 'Excellent': 5}
    df_house['Condition of the House'] = df_house['Condition of the House'].map(condition_map)
    
    # Selecting relevant columns
    house_features = [
        'No of Bedrooms', 'No of Bathrooms', 'Flat Area (in Sqft)', 
        'Lot Area (in Sqft)', 'Condition of the House', 'Overall Grade', 'Zipcode'
    ]
    target_house = 'Sale Price'
    
    # Fill missing values before selection
    for col in house_features + [target_house]:
        df_house[col] = pd.to_numeric(df_house[col], errors='coerce')
        df_house[col] = df_house[col].fillna(df_house[col].median())
    
    X_h = df_house[house_features]
    y_h = df_house[target_house]
    
    house_model = LinearRegression()
    house_model.fit(X_h, y_h)
    
    joblib.dump(house_model, os.path.join(MODEL_DIR, 'house_price_model.pkl'))
    print("House Price Model saved successfully.")
except Exception as e:
    import traceback
    print(f"Error training House model: {e}")
    traceback.print_exc()

print("\n--- Training Loan Eligibility Model ---")
try:
    df_loan = pd.read_csv(LOAN_CSV)
    
    # Mapping Categorical Variables
    df_loan['Married'] = df_loan['Married'].map({'Yes': 1, 'No': 0})
    df_loan['Education'] = df_loan['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df_loan['Property_Area'] = df_loan['Property_Area'].map({'Urban': 0, 'Semiurban': 1, 'Rural': 2})
    df_loan['Loan_Status'] = df_loan['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Features requested in requirements: 
    # Applicant Income, Coapplicant Income, Loan Amount, Loan Term, Credit History, Property Area, Married, Education
    loan_features = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 
        'Married', 'Education'
    ]
    
    # Fill missing values
    for col in loan_features:
        # Use mode[0] safely
        mode_val = df_loan[col].mode()
        if not mode_val.empty:
            df_loan[col] = df_loan[col].fillna(mode_val[0])
        else:
            df_loan[col] = df_loan[col].fillna(0)
    
    # Ensure all features are numeric
    X_l = df_loan[loan_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    y_l = df_loan['Loan_Status'].fillna(0)
    
    loan_model = LogisticRegression(max_iter=5000)
    loan_model.fit(X_l, y_l)
    
    joblib.dump(loan_model, os.path.join(MODEL_DIR, 'loan_eligibility_model.pkl'))
    print("Loan Eligibility Model saved successfully.")
except Exception as e:
    import traceback
    print(f"Error training Loan model: {e}")
    traceback.print_exc()
