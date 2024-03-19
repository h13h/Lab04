import numpy as np
import pickle
from flask import Flask, request, render_template

# Create Flask app instance
app = Flask(__name__)

# Load the trained model from the pickle file
classifier = pickle.load(open("classifier.pkl", "rb"))

# Define route for the home page
@app.route('/')
def home():
    # Render the HTML template for the home page
    return render_template('index.html')

# Define route for the prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the form
    int_features = [float(x) for x in request.form.values()]
    # Convert input features to numpy array
    final_features = [np.array(int_features)]
    # Make prediction using the loaded model
    prediction = classifier.predict(final_features)
    # Render the HTML template with the prediction result
    return render_template('index.html', prediction_text=f'The fish belong to species {prediction[0]}')

# Run the Flask app
if __name__ == '__main__':
    app.run()
