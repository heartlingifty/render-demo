from flask import Flask, request, render_template, jsonify
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# Initialize Flask app
app = Flask(__name__)

# Data for different employee size bands
data_dict = {
    '1': [187600, 180000, 174900, 162500, 158400, 155600, 156700, 152300, 146200, 143100, 138700, 134300, 135800, 131000],
    '2-4': [604700, 599300, 616600, 594700, 643200, 661100, 683100, 715300, 734100, 751200, 756800, 765000, 780400, 773100],
    # (Remaining data...)
}

def fit_and_predict(employee_band, employee_data):
    # Prepare DataFrame
    df_prophet = pd.DataFrame({
        'ds': pd.date_range(start='2010-01-01', periods=len(employee_data), freq='Y'),
        'y': employee_data
    })

    model = Prophet()
    model.fit(df_prophet)

    # Generate future predictions
    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        employee_band = request.form['employee_band']
        if employee_band not in data_dict:
            return render_template('index.html', prediction_text="Invalid Employee Band!")

        # Perform prediction
        forecast_result = fit_and_predict(employee_band, data_dict[employee_band])

        # Get the last 5 years of predictions
        predictions = forecast_result.tail(5).to_dict(orient='records')
        return render_template('index.html', prediction_text=f"Forecast for Employee Band {employee_band}: {predictions}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
