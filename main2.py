import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


# Параметры возмущения
F = 0.5  # Амплитуда возмущения
phi = 0.2  # Фаза возмущения

w_matrix = 1/2 * np.array([
    [-phi, -F],
    [-F, phi]
])

start_proba = np.array([1, 0])  # Начинаем с основного состояния

# Временной диапазон
t_start = 0
t_end = 10 * np.pi # Достаточно большое время для наблюдения динамики
t_points = 1000
t = np.linspace(t_start, t_end, t_points)

probability = np.zeros(t_points, dtype=complex)  # Вероятность остаться в основном состоянии

# print(abs((expm(1j * w_matrix * 20) @ start_proba)[0])**2)

for iv, time in enumerate(t):
    probability[iv] = abs((expm(1j * w_matrix * time) @ start_proba)[0])**2

# Выбираем состояния для отображения (основное и несколько возбужденных)
states = [0, 1, 2]

# Строим графики для каждого состояния
plt.figure(figsize=(10, 6))

plt.plot(t, probability, label='Состояние |0}>')

P = [1 - p for p in probability]
plt.plot(t, P, label='Состояние |1}>')

plt.title('Зависимость вероятности от времени при резонансном возмущении')
plt.xlabel('Время')
plt.ylabel('Вероятность')
plt.legend()
plt.grid(True)
plt.show()