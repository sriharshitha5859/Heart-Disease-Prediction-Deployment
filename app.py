# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from prediction import *

# Load the Random Forest Classifier model
filename = 'heart-disease-rf.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        age = int(request.form['age'])
        sex = int(request.form.get('sex'))
        cp = int(request.form.get('cp'))
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form.get('fbs'))
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form.get('exang'))
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form.get('slope'))
        ca = int(request.form['ca'])
        thal = int(request.form.get('thal'))

        # Check for invalid inputs like '----select option----'
        if sex == -1 or cp == -1 or fbs == -1 or exang == -1 or slope == -1 or thal == -1:
            return render_template('result.html', prediction='Invalid input. Please select valid options.')

        # Make prediction
        print('Check')
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        print(data)
        #feature scaling
        x_app_scaler= scaler.transform(data)
        #print(x_test_scaler)
        
        my_prediction = model.predict(x_app_scaler)[0]
        print(my_prediction)

        # Display the result
        if my_prediction == 0:
            result_text = 'No Problem'
        
        elif my_prediction == 1:
            result_text='Low'
        elif my_prediction == 2:
            result_text='Borderline'
        elif my_prediction == 3:
            result_text = 'Intermediate'
                    
        else:
            result_text = 'High'

        return render_template('result.html', prediction=result_text)
    
    except ValueError as e:
        # Handle value errors (e.g., string to int/float conversion errors)
        return render_template('result.html', prediction='Invalid input data: ' + str(e))
    
    except Exception as e:
        # Catch any other exceptions and show an error message
        return render_template('result.html', prediction='An error occurred: ' + str(e))

if __name__ == '__main__':
    app.run(debug=True)
