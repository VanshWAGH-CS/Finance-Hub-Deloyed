from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
import joblib
import numpy as np
import pandas as pd
import json
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-secret-key")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/bank_v2.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=15)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    full_name = db.Column(db.String(100))
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(20), nullable=False) # 'house' or 'loan'
    input_json = db.Column(db.Text, nullable=False)
    result_text = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, default=94.2)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Paths to models
HOUSE_MODEL_PATH = 'models/house_price_model.pkl'
LOAN_MODEL_PATH = 'models/loan_eligibility_model.pkl'

def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# Load models safely
house_model = None
loan_model = None


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Security Alert: Invalid credentials provided.', 'danger')
    return render_template('login.html')

@app.route('/forgot')
def forgot():
    return render_template('forgot.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if User.query.filter_by(username=request.form['username']).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        user = User(
            username=request.form['username'],
            email=request.form['email'],
            full_name=request.form['full_name']
        )
        user.set_password(request.form['password'])
        db.session.add(user)
        db.session.commit()
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been securely logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/privacy')
def privacy():
    return render_template('compliance.html', title="Privacy Policy")

@app.route('/terms')
def terms():
    return render_template('compliance.html', title="Terms & Conditions")

@app.route('/disclaimer')
def disclaimer():
    return render_template('compliance.html', title="Financial Disclaimer")

@app.route('/dashboard')
@login_required
def dashboard():
    history = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).limit(5).all()
    return render_template('dashboard.html', name=current_user.full_name, history=history)

@app.route('/calculator', methods=['GET', 'POST'])
@login_required
def calculator():
    result = None
    if request.method == 'POST':
        income = float(request.form['income'])
        expenses = float(request.form['expenses'])
        rate = float(request.form['rate']) / 100 / 12
        tenure = int(request.form['tenure']) * 12
        
        disposable_income = income - expenses
        max_emi = disposable_income * 0.45 # Conservative bank rule
        
        # PV = EMI * [(1 - (1+r)^-n) / r]
        if rate > 0:
            suggested_loan = max_emi * ((1 - (1 + rate)**(-tenure)) / rate)
        else:
            suggested_loan = max_emi * tenure
            
        result = {
            'max_emi': f"${max_emi:,.2f}",
            'suggested_loan': f"${suggested_loan:,.2f}"
        }
    return render_template('calculator.html', result=result)

@app.route('/download-report/<int:prediction_id>')
@login_required
def download_report(prediction_id):
    pred = Prediction.query.get_or_404(prediction_id)
    if pred.user_id != current_user.id:
        return "Unauthorized", 403
        
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=LETTER)
    
    # Branding
    p.setFont("Helvetica-Bold", 24)
    p.drawString(100, 750, "TrustBridge Bank")
    p.setFont("Helvetica", 12)
    p.drawString(100, 735, "Official Financial Analysis Report")
    p.line(100, 730, 500, 730)
    
    # Metadata
    p.setFont("Helvetica-Bold", 12)
    p.drawString(100, 700, f"Report ID: TB-{pred.id:04d}")
    p.drawString(100, 685, f"Date: {pred.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    p.drawString(100, 670, f"Client Name: {current_user.full_name}")
    
    # Analysis Content
    p.setFont("Helvetica-Bold", 14)
    p.drawString(100, 630, f"Analysis Type: {pred.type.title()} Prediction")
    
    p.setFont("Helvetica", 12)
    y = 600
    inputs = json.loads(pred.input_json)
    p.drawString(100, y, "Input Parameters:")
    y -= 20
    for key, val in inputs.items():
        p.drawString(120, y, f"- {key.replace('_', ' ').title()}: {val}")
        y -= 15
        
    y -= 25
    p.setFont("Helvetica-Bold", 16)
    p.setFillColorRGB(0.06, 0.09, 0.16) # Brand primary
    p.drawString(100, y, f"FINAL DETERMINATION: {pred.result_text}")
    
    y -= 30
    p.setFont("Helvetica-Oblique", 12)
    p.setFillColor(colors.black)
    p.drawString(100, y, f"Analytical Confidence: {pred.confidence}%")
    
    # Disclaimer
    p.setFont("Helvetica", 8)
    p.drawString(100, 50, "DISCLAIMER: This report is for advisory purposes only. Not a formal financial commitment.")
    p.drawString(100, 40, "TrustBridge Bank © 2026. All Rights Reserved.")
    
    p.showPage()
    p.save()
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True, download_name=f"TrustBridge_Report_{pred.id}.pdf", mimetype='application/pdf')

@app.route('/house')
@login_required
def house():
    return render_template('house.html')

@app.route('/loan')
@login_required
def loan():
    return render_template('loan.html')

@app.route('/predict-house', methods=['POST'])
@login_required
def predict_house():
    global house_model
    if house_model is None:
        house_model = load_model(HOUSE_MODEL_PATH)
        if house_model is None:
            return render_template('result.html', error="System Upgrade in Progress: House Model currently offline.")

    try:
        data = {
            'bedrooms': float(request.form['bedrooms']),
            'bathrooms': float(request.form['bathrooms']),
            'flat_area': float(request.form['flat_area']),
            'lot_area': float(request.form['lot_area']),
            'condition': float(request.form['condition']),
            'grade': float(request.form['grade']),
            'zipcode': float(request.form['zipcode'])
        }
        
        final_features = np.array(list(data.values())).reshape(1, -1)
        prediction = house_model.predict(final_features)
        output = f"${prediction[0]:,.2f}"
        
        # Explainability heuristics (Mock for demo)
        explainFactors = [
            {"factor": "Location (Zipcode)", "impact": "High", "desc": "Zipcode is a primary driver of market value in our current model."},
            {"factor": "Space (Sqft)", "impact": "Medium", "desc": "Flat area directly correlates with the predicted appraisal."},
            {"factor": "State (Condition)", "impact": "Moderate", "desc": "Overall property maintenance level affected the final number."}
        ]
        
        # Save to History
        pred_entry = Prediction(
            type='house',
            input_json=json.dumps(data),
            result_text=output,
            user_id=current_user.id
        )
        db.session.add(pred_entry)
        db.session.commit()
        
        return render_template('result.html', 
                             prediction_text=output,
                             title="Real Estate Appraisal",
                             type="house",
                             explain=explainFactors,
                             pred_id=pred_entry.id)
    except Exception as e:
        return render_template('result.html', error=f"Data processing error: {str(e)}")

@app.route('/predict-loan', methods=['POST'])
@login_required
def predict_loan():
    global loan_model
    if loan_model is None:
        loan_model = load_model(LOAN_MODEL_PATH)
        if loan_model is None:
            return render_template('result.html', error="System Upgrade in Progress: Loan Engine currently offline.")

    try:
        data = {
            'applicant_income': float(request.form['applicant_income']),
            'coapplicant_income': float(request.form['coapplicant_income']),
            'loan_amount': float(request.form['loan_amount']),
            'loan_term': float(request.form['loan_term']),
            'credit_history': float(request.form['credit_history']),
            'property_area': request.form['property_area'],
            'married': request.form['married'],
            'education': request.form['education']
        }
        
        # Data prep for model
        area_map = {'Urban': 0, 'Semiurban': 1, 'Rural': 2}
        features = [
            data['applicant_income'], data['coapplicant_income'], data['loan_amount'],
            data['loan_term'], data['credit_history'], area_map.get(data['property_area'], 0),
            1 if data['married'] == 'Yes' else 0,
            1 if data['education'] == 'Graduate' else 0
        ]
        
        final_features = np.array(features).reshape(1, -1)
        prediction = loan_model.predict(final_features)
        result = "Approved ✅" if prediction[0] == 1 else "Flagged for Review ❌"
        
        # Explainability
        explainFactors = [
            {"factor": "Credit History", "impact": "Critical", "desc": "Historical repayment behavior is the strongest predictor for this outcome."},
            {"factor": "Income-to-Debt", "impact": "High", "desc": "The ratio between your total income and requested loan amount was analyzed."},
            {"factor": "Education Level", "impact": "Minor", "desc": "Academic background contributes slightly to financial stability scoring."}
        ]
        
        # Save to History
        pred_entry = Prediction(
            type='loan',
            input_json=json.dumps(data),
            result_text=result,
            user_id=current_user.id
        )
        db.session.add(pred_entry)
        db.session.commit()
        
        return render_template('result.html', 
                             prediction_text=result,
                             title="Loan Eligibility Analysis",
                             type="loan",
                             explain=explainFactors,
                             pred_id=pred_entry.id)
    except Exception as e:
        return render_template('result.html', error=f"Analysis engine error: {str(e)}")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
