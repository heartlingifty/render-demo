from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the business predictions data
predictions_path = 'business_prediction.pkl'

# Attempt to load pickle file with forecast data
try:
    with open(predictions_path, 'rb') as file:
        forecast_result = pickle.load(file)
    print("Forecast data loaded successfully.")
except EOFError:
    print(f"Error: The pickle file {predictions_path} is empty or corrupted.")
    forecast_result = {}  # Empty data if file is corrupted
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
    forecast_result = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Extract form data
        industry = request.form['industry']  # Industry key from the form
        year = int(request.form['year'])    # Year input
        
        # Retrieve prediction for the given industry and year
        if industry in forecast_result and year in forecast_result[industry]:
            forecast_value = forecast_result[industry][year]
            message = f"The forecasted business number for '{industry}' in {year} is {forecast_value}."
        else:
            message = "No forecast data available for the specified industry and year."
    except Exception as e:
        message = f"Error: {str(e)}"
    
    return render_template('index.html', prediction_text=message)

if __name__ == "__main__":
    app.run(debug=True)
