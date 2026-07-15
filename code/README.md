# Smart Lender – Loan Eligibility Prediction using Machine Learning

## Overview

Smart Lender is a Machine Learning-based web application that predicts whether a loan application is likely to be approved based on applicant details. The system helps financial institutions make faster, more consistent, and data-driven lending decisions.

The application is developed using Python, Flask, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, and Seaborn.

---

## Features

- Loan Eligibility Prediction
- Machine Learning-Based Decision Support
- User-Friendly Flask Web Interface
- Real-Time Prediction
- Data Visualization
- Model Comparison
- Confusion Matrix
- Feature Importance Analysis
- Correlation Heatmap

---

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- HTML
- CSS
- Bootstrap

---

## Machine Learning Models

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- XGBoost

After comparison, the best-performing model is automatically selected and saved for prediction.

---

## Dataset

- Records: 614
- Features: 12 Input Features
- Target:
  - Loan Approved
  - Loan Rejected

Input features include:

- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area

---

## Installation

```bash
pip install -r requirements.txt
python model.py
python app.py
```

---

## Project Structure

```
smart_lender/
│
├── app.py
├── model.py
├── dataset.csv
├── requirements.txt
├── README.md
│
├── models/
│   └── loan_model.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── images/
│
└── screenshots/
```

---

## Results

Best Model: Random Forest

Testing Accuracy: **82.11%**

---

## Future Enhancements

- Cloud Deployment
- Deep Learning Models
- Loan Risk Score
- Explainable AI
- Admin Dashboard
- Database Integration

---

## Developed By

Smart Lender Team

2026