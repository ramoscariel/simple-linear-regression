import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Cargar datos
dataset = pd.read_csv('Student_Marks.csv')

# Definir variable Independiente & dependiente
X = dataset[['time_study']]
y = dataset['Marks']

# Dividir dataset en conjuntos: entrenamiento (2/3) & prueba (1/3)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Entrenar modelo SLR
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

# Resultados Entrenamiento
plt.scatter(X_train, y_train, color='red') # Puntos reales en el conjunto de entrenamiento
plt.plot(X_train, model.predict(X_train), color='blue') # Línea de regresión ajustada
plt.title('Puntaje vs Tiempo de estudio (Conjunto de Entrenamiento)')
plt.xlabel('Tiempo de estudio')
plt.ylabel('Puntaje')
plt.show()

# Resultados Prueba
plt.scatter(X_test, y_test, color='red') # Puntos reales en el conjunto de prueba
plt.plot(X_train, model.predict(X_train), color='blue') # Línea de regresión ajustada (misma que en entrenamiento)
plt.title('Puntaje vs Tiempo de estudio (Conjunto de Prueba)')
plt.xlabel('Tiempo de estudio')
plt.ylabel('Puntaje')
plt.show()

# Evaluación
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test) # predicciones en base a los datos de prueba
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² score:", r2_score(y_test, y_pred))