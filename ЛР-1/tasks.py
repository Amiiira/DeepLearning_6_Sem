from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import numpy as np
from main import data


#кодируем бинарные категориальные признаки
data['smoker'] = data['smoker'].apply(lambda x: 0 if x == 'no' else 1)
data['sex'] = data['sex'].apply(lambda x: 0 if x == 'female' else 1)
#оставшиеся категориальные признаки кодируем с помощью OneHot
data = pd.get_dummies(data)
data.head()

"""  
    Задание 1: 
"""
data = pd.read_csv('data/insurance.csv')
X = data.drop(columns=['charges'])  # признаки
y = data['charges']  # целевая переменная

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Вывод размеров полученных выборок для проверки
print("Размеры тренировочной выборки (X_train, y_train):", X_train.shape, y_train.shape)
print("Размеры тестовой выборки (X_test, y_test):", X_test.shape, y_test.shape)
print("Количество наблюдений в тестовом наборе данных:", X_test.shape[0])

"""  
    Задание 2: 
"""

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Получение свободного члена (intercept)
intercept = round(model.intercept_, 2)
print("Свободный член (intercept):", intercept)

# Предсказания на тренировочной и тестовой выборках
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Рассчет метрик качества на тренировочной выборке
r2_train = round(r2_score(y_train, y_train_pred), 3)
mae_train = round(mean_absolute_error(y_train, y_train_pred))
mape_train = round(mean_absolute_percentage_error(y_train, y_train_pred) * 100)

# Рассчет метрик качества на тестовой выборке
r2_test = round(r2_score(y_test, y_test_pred), 3)
mae_test = round(mean_absolute_error(y_test, y_test_pred))
mape_test = round(mean_absolute_percentage_error(y_test, y_test_pred) * 100)

# Вывод результатов
print("\nМетрики на тренировочной выборке:")
print("R^2:", r2_train)
print("MAE:", mae_train)
print("MAPE:", mape_train, "%")

print("\nМетрики на тестовой выборке:")
print("R^2:", r2_test)
print("MAE:", mae_test)
print("MAPE:", mape_test, "%")

"""  
    Задание 3: 
"""
# Вычисление ошибок на тренировочной и тестовой выборках
train_errors = y_train - y_train_pred
test_errors = y_test - y_test_pred

# Построение диаграммы boxplot
plt.figure(figsize=(10, 6))
plt.boxplot([train_errors, test_errors], labels=['Тренировочная выборка', 'Тестовая выборка'])
plt.title('Диаграмма boxplot ошибок модели линейной регрессии')
plt.xlabel('Выборка')
plt.ylabel('Ошибки (y - y_pred)')
plt.grid(True)
plt.show()

# Анализ ответов
# Ответы:
# A - Разброс ошибок на тестовой выборке больше, чем на тренировочной
# B - Разброс ошибок на тренировочной выборке больше, чем на тестовой
# C - Медианная ошибка на тренировочной и тестовой выборках отрицательная (меньше 0)
# D - Медианная ошибка на тренировочной и тестовой выборках положительная (больше 0)

# Определим, какие из ответов верны
median_train_error = train_errors.median()
median_test_error = test_errors.median()

answer_A = True if test_errors.std() > train_errors.std() else False
answer_B = True if train_errors.std() > test_errors.std() else False
answer_C = True if median_train_error < 0 and median_test_error < 0 else False
answer_D = True if median_train_error > 0 and median_test_error > 0 else False

print("A:", answer_A)
print("B:", answer_B)
print("C:", answer_C)
print("D:", answer_D)

"""  
    Задание 4: 
"""

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Генерация полиномиальных признаков степени 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print("Размеры нормализованных и полиномиальных данных:")
print("X_train_poly:", X_train_poly.shape)
print("X_test_poly:", X_test_poly.shape)

"""  
    Задание 5: 
"""

# Обучение модели линейной регрессии на полиномиальных признаках
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# Предсказание на тестовой выборке
y_test_pred_poly = model_poly.predict(X_test_poly)

# Вычисление метрики R^2 на тестовой выборке
r2_test_poly = r2_score(y_test, y_test_pred_poly)

# Вывод значения метрики R^2
print("Значение метрики R^2 на тестовой выборке:", round(r2_test_poly, 3))

"""  
    Задание 6: 
"""

# Вывод значений коэффициентов
coefficients = model_poly.coef_
print("Значения коэффициентов:", coefficients)

# Вывод степеней коэффициентов
powers = poly.powers_
print("Степени коэффициентов:", powers)

"""  
    Задание 7: 
"""

# Обучение модели Lasso на полиномиальных признаках
model_lasso = Lasso(max_iter=2000)
model_lasso.fit(X_train_poly, y_train)

# Предсказание на тестовой выборке
y_test_pred_lasso = model_lasso.predict(X_test_poly)

# Вычисление метрики R^2 на тестовой выборке
r2_test_lasso = r2_score(y_test, y_test_pred_lasso)

# Вычисление метрики MAE на тестовой выборке
mae_test_lasso = mean_absolute_error(y_test, y_test_pred_lasso)

# Вычисление метрики MAPE на тестовой выборке
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_test_lasso = mean_absolute_percentage_error(y_test, y_test_pred_lasso)

# Вывод значений метрик
print("Метрики на тестовой выборке для модели Lasso:")
print("R^2:", round(r2_test_lasso, 3))
print("MAE:", round(mae_test_lasso))
print("MAPE:", round(mape_test_lasso))