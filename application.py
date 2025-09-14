from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        # Create a DataFrame for the input data
        input_data = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]],
                                   columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region'])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make predictions
        prediction = ridge_model.predict(input_data_scaled)

        return render_template('home.html', result=prediction[0])

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)
