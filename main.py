from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Crear la app
app = FastAPI(title="Heart Disease Prediction API")

# Cargar el modelo entrenado
modelo = joblib.load("modeloEntrenado.pkl")

# Definir la estructura de los datos de entrada
class HeartData(BaseModel):
    age: int
    male: int
    sysBP: float
    totChol: float
    currentSmoker: int
    diabetes: int
    BMI: float

# Endpoint raíz
@app.get("/")
def read_root():
    return {"message": "API de predicción de enfermedad cardíaca"}

# Endpoint para predecir
@app.post("/predict")
def predict(data: HeartData):
    # Convertir los datos de entrada en DataFrame
    df = pd.DataFrame([data.model_dump()])  # lista con un dict

    # Hacer la predicción
    prob = modelo.predict_proba(df)[:, 1][0]  # probabilidad de enfermedad
    pred = modelo.predict(df)[0]             # predicción binaria

    return {"prediction": int(pred), "probability": float(prob)}
