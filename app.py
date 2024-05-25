
from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)


model=pickle.load(open("Winemodel.pkl",'rb'))


car=pd.read_csv("winequality-red.csv")


@app.route('/')
def index():
    return render_template('index.html')

#     return render_template('index.html', Brand= Brand, Fuel_type= Fuel_type,Aspiration=Aspiration,Door_Panel=Door_Panel,Design=Design,Wheel_Drive=Wheel_Drive,Engine_Location=Engine_Location,Engine_Type=Engine_Type,Cylinder_Count=Cylinder_Count,Fuel_System=Fuel_System)

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from the form
    fixed_acidity = float(request.form.get('user'))  
    volatile_acidity = float(request.form.get('week'))  
    citric_acid = float(request.form.get('hour'))  
    residual_sugar = float(request.form.get('age'))  
    chlorides = float(request.form.get('screen'))  
    free_sulfur_dioxide = int(request.form.get('game'))  
    total_sulfur_dioxide = int(request.form.get('feature'))  
    density = float(request.form.get('location')) 
    pH = float(request.form.get('ph'))  
    sulphates = float(request.form.get('sulphates'))  
    alcohol = float(request.form.get('alcohol'))  
    

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })

    # Predict using the model
    prediction = model.predict(input_data)

    # Return the prediction
    return str(prediction[0])



if __name__=="__main__":
    app.run(debug=True)