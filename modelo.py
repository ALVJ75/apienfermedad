import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('framingham.csv')

# print(df.shape)

# print(df.info())

# print(df.isna().sum())

# Limpiamos el dataframe para que no tenga nulos
df = df.dropna(subset=["BMI", "totChol"])

# Definimos con que columnas vamos a trabajar
input_col = ["age","male","sysBP","totChol","currentSmoker","diabetes","BMI"]
# Definimos la columna que despues queremos predecir
objetivo = 'TenYearCHD'

# Esto fue para revisar que no existan nulos en las columnas que vamos a usar
# print(df[input_col].isna().sum())
# print(df[input_col].shape)
# print(df[input_col].info())

# Generamos los ejes de 'x' y 'y'

x = df[input_col] # x hace referencia a los datos que tenemos que poner para que nos de la prediccion (columnas predictorias)

y = df[objetivo] # y es la variable que queremos predecir 

# Ahora vamos a dividir todo el dataframe en un 80 20, para despues realizar el entrenamiento del modelo

x_train, x_test, y_train, y_test = train_test_split(
    x, y, #Colocamos las variables que definimos
    test_size=0.2, # definimos cuanto es la division, diciendo test_size=0.2 decimos que test sea del 20% de los datos
    random_state=42, #Es una forma de fijar el proceso aleatorio, es como una semilla del maincra, te fija el mundo
    stratify=y #Separa la proporcion de positivos y negativos equitativamente
)

# Ahora creamos el modelo
modelo = LogisticRegression(max_iter=1000) #el max_iter es definir el numero maximo de iteraciones que tiene el modelo para aprender los parametros

# Ahora lo entreamos con los datos de train
modelo.fit(x_train, y_train)

# Ahora vamos a exportar el modelo para despues usarlo en la API
joblib.dump(modelo, "modeloEntrenado.pkl")

# Aqui ya hacemos las predicciones
y_pred = modelo.predict(x_test)

# Aqui vemos que tan exacto fue (las metricas)
tp, fp, fn, tn = confusion_matrix(y_test, y_pred).ravel()

print("Verdaderos Positivos (TP):", tp)
print("Verdaderos Negativos (TN):", tn)
print("Falsos Positivos (FP):", fp)
print("Falsos Negativos (FN):", fn)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("F1:", f1)

# Probabilidad de clase 1
y_prob = modelo.predict_proba(x_test)[:, 1]

# Calcular FPR y TPR
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calcular ROC AUC
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", roc_auc)