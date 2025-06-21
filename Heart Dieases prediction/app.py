from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset and preprocess
heart_data = pd.read_csv('heart.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Initialize Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form and convert to numpy array
    input_data = [
        int(request.form['age']),
        int(request.form['sex']),
        int(request.form['cp']),
        int(request.form['trestbps']),
        int(request.form['chol']),
        int(request.form['fbs']),
        int(request.form['restecg']),
        int(request.form['thalach']),
        int(request.form['exang']),
        float(request.form['oldpeak']),
        int(request.form['slope']),
        int(request.form['ca']),
        int(request.form['thal']),
    ]
    
    # Convert input data to numpy array and reshape for prediction
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_as_numpy_array)
    
    # Determine prediction outcome
    result = 'The Person has Heart Disease' if prediction[0] == 1 else 'The Person does not have a Heart Disease'
    return render_template('result.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
