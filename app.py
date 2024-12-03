from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved ML model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    features = [float(x) for x in request.form.values()]
    features_array = np.array(features).reshape(1, -1)  # Reshape for prediction
    prediction = model.predict(features_array)  # Get prediction

    return render_template(
        'index.html',
        prediction_text=f'Predicted Value: {prediction[0]:.2f}'
    )

if __name__ == "__main__":
    app.run(debug=True)
