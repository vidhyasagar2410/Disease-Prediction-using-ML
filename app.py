
from flask import Flask, render_template, request

# Load the dataset containing symptom-disease mappings
# Assuming the dataset is in a CSV file format
import pandas as pd
import numpy as np
import pickle

final_rf_model = pickle.load(open('model.pkl','rb'))
encoder=pickle.load(open('encoder.pkl','rb'))
df = pd.read_csv('Training.csv')

app = Flask(_

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict(): 
    X=pd.read_csv("X.csv")
    symptoms = X.columns.values


    symptom_index = {}
    for index, value in enumerate(symptoms): 
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index

    data_dict = { 
        "symptom_index": symptom_index,
        "predictions_classes": encoder.classes_
        }

    def predictDisease(symptoms): 
        symptoms = symptoms.split(",")

        input_data = [0] * len(data_dict["symptom_index"])
        for symptom in symptoms: 
            if symptom in data_dict["symptom_index"]: 
                index = data_dict["symptom_index"][symptom]
                input_data[index] = 1

        input_data = np.array(input_data).reshape(1, -1)

        rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]

        return rf_prediction



    # Get the symptoms from the form data
    symptoms = ",".join([request.form['symptom1'], request.form['symptom2'], request.form['symptom3'], request.form['symptom4']])
    
   
    
    return render_template('index.html', y="The disaese you might have is ..  "+predictDisease(symptoms))

if __name__ == '__main__':
    app.run(debug=True)
