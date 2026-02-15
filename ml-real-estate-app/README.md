# AI-Powered Real Estate & Finance Predictor

A production-ready Machine Learning web application built with Python and Flask. This application provides two key functionalities:
1. **House Price Prediction**: Uses regression to estimate property values.
2. **Loan Eligibility Check**: Uses classification to determine loan approval status.

## 🚀 Key Features
- **Modern UI**: Clean, professional responsive design using Bootstrap 5.
- **Glassmorphism**: Elegant aesthetics with blurred backgrounds and smooth gradients.
- **Robust Backend**: Flask-based API with error handling for missing models.
- **Easy Deployment**: Fully compatible with Replit and other cloud platforms.

## 📁 Project Structure
```
ml-real-estate-app/
│
├── models/
│   ├── house_price_model.pkl      # Upload your trained house model here
│   └── loan_eligibility_model.pkl # Upload your trained loan model here
│
├── app.py                         # Main Flask application
├── requirements.txt               # Dependencies
│
├── templates/
│   ├── index.html                 # Home page
│   ├── house.html                 # House price input form
│   ├── loan.html                  # Loan eligibility input form
│   └── result.html                # Prediction result page
│
├── static/
│   └── style.css                  # Custom premium styles
│
└── README.md                      # Documentation
```

## 🛠️ Installation & Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Upload Models**:
   Place your `.pkl` files inside the `models/` directory.
3. **Run Application**:
   ```bash
   python app.py
   ```
   The app will be available at `http://localhost:5000` (or the port defined in your environment).

## 📊 Model Expectation
### House Price Model
- **Inputs**: Bedrooms, Bathrooms, Flat Area (sqft), Lot Area (sqft), Condition (1-5), Grade (1-13), Zipcode.
- **Output**: Numerical price value.

### Loan Eligibility Model
- **Inputs**: Applicant Income, Coapplicant Income, Loan Amount, Loan Term, Credit History, Property Area (Urban/Semiurban/Rural), Married (Yes/No), Education (Grad/Not Grad).
- **Output**: Binary (1 for Approved, 0 for Rejected).

## ⚖️ License
MIT License
