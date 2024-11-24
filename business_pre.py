import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Data for different employee size bands
data_dict = {
    '1': [187600, 180000, 174900, 162500, 158400, 155600, 156700, 152300,
          146200, 143100, 138700, 134300, 135800, 131000],
    '2-4': [604700, 599300, 616600, 594700, 643200, 661100, 683100, 715300,
            734100, 751200, 756800, 765000, 780400, 773100],
    '5-9': [223000, 220700, 231200, 229800, 242800, 252200, 241600, 250200,
            257000, 261000, 261400, 262900, 270800, 273200],
    '10-19': [112800, 110300, 117600, 121500, 127300, 133600, 133400, 135700,
               137400, 138300, 138000, 136900, 141700, 144300],
    '20-49': [60800, 64000, 60400, 65200, 67400, 69900, 70100, 72200,
              72200, 73000, 73800, 73700, 75600, 78500],
    '50-99': [19300, 20400, 19600, 20200, 20800, 21500, 22000, 22400,
              23000, 23500, 23800, 23500, 23800, 24500],
    '100-199': [8300, 8600, 8500, 8800, 8900, 9200, 9400, 9600,
                9800, 10000, 10200, 10100, 10100, 10300],
    '200-249': [1600, 1700, 1700, 1700, 1800, 1900, 1900, 1900,
                2000, 2100, 2100, 2000, 2000, 2100],
    '250-499': [3200, 3200, 3300, 3400, 3500, 3600, 3700, 3700,
                3800, 3900, 4000, 3900, 3900, 4100],
    '500 or more': [3100, 3100, 3100, 3200, 3300, 3400, 3500, 3500,
                    3700, 3800, 3800, 3800, 3800, 3900],
}

# Function to fit Prophet model and plot predictions
def fit_and_predict(employee_band, employee_data):
    # Create DataFrame for the current employee band
    df_prophet = pd.DataFrame({
        'ds': pd.date_range(start='2010-01-01', periods=len(employee_data), freq='Y'),
        'y': employee_data
    })

    # Drop any missing data
    df_prophet = df_prophet.dropna(subset=['y'])

    # Fit the Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Create future dates for prediction (predict for 5 years into the future)
    future = model.make_future_dataframe(periods=5, freq='Y')

    # Generate forecast
    forecast = model.predict(future)

    # Plot the historical data and the forecast
    fig = model.plot(forecast)
    plt.title(f'Business Count Prediction for Employee Size Band: {employee_band}')
    plt.xlabel('Year')
    plt.ylabel('Business Count')
    plt.show()

    # Return forecast data for inspection
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Run predictions for each employee size band
for band, data in data_dict.items():
    forecast_result = fit_and_predict(band, data)
    print(f'Forecast for employee size band {band}:')
    print(forecast_result.tail())