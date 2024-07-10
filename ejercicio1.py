import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sympy as sp

# Datos proporcionados
x_data = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3, 6.8, 7.1])
y_data = np.array([102.56, 130.11, 113.18, 142.05, 167.53,
                  195.14, 224.87, 256.73, 299.50, 326.72])

# Función para calcular el error cuadrático medio


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Ajustar y calcular el error para polinomios de grados 1, 2 y 3
degrees = [1, 2, 3]
coeffs = []
errors = []
polynomials = []
x = sp.symbols('x')

for degree in degrees:
    coeff = np.polyfit(x_data, y_data, degree)
    p = np.poly1d(coeff)
    y_pred = p(x_data)
    error = mse(y_data, y_pred)
    coeffs.append(coeff)
    errors.append(error)

    # Crear el polinomio simbólico
    poly_expr = sum(c * x**i for i, c in enumerate(reversed(coeff)))
    polynomials.append(poly_expr)

# Ajustar y calcular el error para la forma b e^(ax)


def exp_func(x, a, b):
    return b * np.exp(a * x)


params_exp, _ = curve_fit(exp_func, x_data, y_data)
y_pred_exp = exp_func(x_data, *params_exp)
error_exp = mse(y_data, y_pred_exp)

# Crear la función simbólica para b e^(ax)
a_exp, b_exp = params_exp
exp_expr = b_exp * sp.exp(a_exp * x)

# Ajustar y calcular el error para la forma b x^a


def power_func(x, a, b):
    return b * x ** a


params_power, _ = curve_fit(power_func, x_data, y_data)
y_pred_power = power_func(x_data, *params_power)
error_power = mse(y_data, y_pred_power)

# Crear la función simbólica para b x^a
a_power, b_power = params_power
power_expr = b_power * x**a_power

# Imprimir resultados
for i, degree in enumerate(degrees):
    print(f"Polinomio de grado {degree}: {
          polynomials[i]}, error = {errors[i]}")

print(f"Forma b e^(ax): {exp_expr}, error = {error_exp}")
print(f"Forma b x^a: {power_expr}, error = {error_power}")

# Función para graficar y mostrar la imagen


def plot_and_show(x_data, y_data, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Datos')
    plt.plot(x_data, y_pred, label=title)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.show()


# Graficar y mostrar las imágenes
plot_and_show(x_data, y_data, np.poly1d(
    coeffs[0])(x_data), 'Ajuste Polinomio Grado 1')
plot_and_show(x_data, y_data, np.poly1d(
    coeffs[1])(x_data), 'Ajuste Polinomio Grado 2')
plot_and_show(x_data, y_data, np.poly1d(
    coeffs[2])(x_data), 'Ajuste Polinomio Grado 3')
plot_and_show(x_data, y_data, exp_func(
    x_data, *params_exp), 'Ajuste Forma b e^(ax)')
plot_and_show(x_data, y_data, power_func(
    x_data, *params_power), 'Ajuste Forma b x^a')
