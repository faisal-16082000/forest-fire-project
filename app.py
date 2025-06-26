from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', result=None)

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        inputs = {}
        for key in ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']:
            val = request.form.get(key)
            if val is None or val == '':
                return render_template('index.html', result=f"Error: Missing input for {key}")
            inputs[key] = val

        Temperature = float(inputs['Temperature'])
        RH = float(inputs['RH'])
        Ws = float(inputs['Ws'])
        Rain = float(inputs['Rain'])
        FFMC = float(inputs['FFMC'])
        DMC = float(inputs['DMC'])
        ISI = float(inputs['ISI'])
        Classes = int(inputs['Classes'])
        Region = int(inputs['Region'])

        new_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        print("Input data:", new_data)

        new_data_scaled = standard_scaler.transform(new_data)
        print("Scaled data:", new_data_scaled)

        result = ridge_model.predict(new_data_scaled)
        print("Prediction:", result)

        return render_template('index.html', result=f"{result[0]:.2f}")
    except Exception as e:
        print("Error during prediction:", e)
        return render_template('index.html', result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
