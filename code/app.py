from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/loan_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "Gender": request.form["Gender"],
        "Married": request.form["Married"],
        "Dependents": request.form["Dependents"],
        "Education": request.form["Education"],
        "Self_Employed": request.form["Self_Employed"],
        "ApplicantIncome": float(request.form["ApplicantIncome"]),
        "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
        "LoanAmount": float(request.form["LoanAmount"]),
        "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
        "Credit_History": float(request.form["Credit_History"]),
        "Property_Area": request.form["Property_Area"]
    }

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    confidence = max(probability) * 100

    if prediction == 1:
        result = "✅ Loan Approved"
        alert = "success"
    else:
        result = "❌ Loan Rejected"
        alert = "danger"

    return render_template(
        "index.html",
        prediction_text=result,
        confidence=f"{confidence:.2f}",
        alert=alert
    )


if __name__ == "__main__":
    app.run(debug=True)