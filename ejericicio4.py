import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Datos de la tabla
weights = [4800, 3700, 3400, 2800, 1900]
percentages = [3.1, 4.0, 5.2, 6.4, 9.6]

# Convertir a arrays de numpy
x = np.array(weights)
y = np.array(percentages)

# Calcular la regresión lineal
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Ecuación de la recta de mínimos cuadrados
line = slope * x + intercept

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Datos')
plt.plot(x, line, color='red', label=f'Recta de mínimos cuadrados: y = {
         slope:.4f}x + {intercept:.2f}')
plt.xlabel('Peso promedio (lb)')
plt.ylabel('Porcentaje de presentación')
plt.title('Regresión Lineal de Peso promedio vs Porcentaje de presentación')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar la ecuación de la recta y el coeficiente de determinación
slope, intercept, r_value**2
