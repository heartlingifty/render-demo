import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
table_4 = pd.read_csv("table_4.csv")  # Ensure 'table_4.csv' is in the same directory

# Preprocessing
table_4.columns = table_4.columns.str.strip().str.lower().str.replace('\n', ' ').str.replace('  ', ' ')
table_4.rename(columns={
    "industry": "Industry",
    "business number": "Business Number",
    "employment": "Employment",
    "turnover (£ millions, excluding vat)": "Turnover (£ millions, excluding VAT)"
}, inplace=True)
table_4["Turnover (£ millions, excluding VAT)"] = pd.to_numeric(
    table_4["Turnover (£ millions, excluding VAT)"], errors='coerce'
)
table_4["Turnover (£ millions, excluding VAT)"].fillna(
    table_4["Turnover (£ millions, excluding VAT)"].mean(), inplace=True
)

# Prepare features and target
X = table_4[["Business Number", "Employment"]]
y = table_4["Turnover (£ millions, excluding VAT)"]

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Define route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Define API route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    industry = data.get("industry")

    # Find the relevant industry data
    industry_data = table_4[table_4["Industry"] == industry]
    if industry_data.empty:
        return jsonify({"error": "Industry not found"})

    # Extract current values
    current_business = industry_data["Business Number"].values[0]
    current_employment = industry_data["Employment"].values[0]

    # Predict for the next 10 years
    predictions = []
    for year in range(1, 11):
        future_business = current_business * (1 + 0.015 * year)
        future_employment = current_employment * (1 + 0.015 * year)
        future_turnover = model.predict([[future_business, future_employment]])[0]
        predictions.append({
            "Year": 2024 + year,
            "Business Number": future_business,
            "Employment": future_employment,
            "Turnover": future_turnover
        })

    return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)

