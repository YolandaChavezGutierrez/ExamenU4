#- Implementa el modelo SVM guardado con FastAPI y Uvicorn.
import uvicorn
from fastapi import FastAPI
from diabetesModel import diabetes
import numpy as np
import pickle
import pandas as pd
#- Levanta un servidor local que exponga un punto final de API para realizar  predicciones utilizando el modelo implementado.

app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)

@app.get("/")
def index():
    return {"mensaje":"Hola bienvenido al modelo"}

@app.get("/Bienvenida")
def fun_nombre(name:str):
    return {"Hola bienvenido":f"{name}"}

@app.post("/predict")
def predict_diabetes(data:diabetes):
    data = data.dict()
    Pregnancies = data["Pregnancies"]
    Glucose = data["Glucose"]
    BloodPressure = data["BloodPressure"]
    SkinThickness = data["SkinThickness"]
    Insulin = data["Insulin"]
    BMI = data["BMI"]
    DiabetesPedigreeFunction = data["DiabetesPedigreeFunction"]
    Age = data["Age"]

    prediction = classifier.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])

    if(prediction[0] > 5):
        prediction = "Probablemente sea diabetico por su edad"
    else:
        prediction = "Probablemente no, basado solo en su edad"
    return {"prediction":prediction}

if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=8000)
#- Pruebe el punto final de la API utilizando datos de entrada de muestra y asegúresede que arroja predicciones precisas.
