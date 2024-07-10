import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Datos de la tabla
act_scores = [28, 25, 28, 27, 28, 33, 28, 29, 23,
              27, 29, 28, 27, 29, 21, 28, 28, 26, 30, 24]
average_scores = [3.84, 3.21, 3.23, 3.63, 3.75, 3.20, 3.41, 3.38, 3.53,
                  2.03, 3.75, 3.65, 3.87, 3.75, 1.66, 3.12, 2.96, 2.92, 3.10, 2.81]

# Convertir a arrays de numpy
x = np.array(act_scores)
y = np.array(average_scores)

# Calcular la regresión lineal
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Ecuación de la recta de mínimos cuadrados
line = slope * x + intercept

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Datos')
plt.plot(x, line, color='red', label=f'Recta de mínimos cuadrados: y = {
         slope:.2f}x + {intercept:.2f}')
plt.xlabel('Puntuación ACT')
plt.ylabel('Promedio de puntos')
plt.title('Regresión Lineal de Puntuación ACT vs Promedio de puntos')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar la ecuación de la recta y el coeficiente de determinación
slope, intercept, r_value**2
