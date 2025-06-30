import numpy as np
from scipy.integrate import odeint

# Параметры системы
F = 1.0          # Амплитуда возмущения
freq = 1.0       # Частота осциллятора
max_level = 100  # Максимальный уровень
hbar = 1.0       # Приведенная постоянная Планка
m = 1.0          # Масса частицы
time = 10        # Время рассмотрения процесса


def count_proba_fermi(state, time):
    '''
    Численное решение уравнения для вероятностей
    
    Parameters:
        state (np.array): Начальные вероятности (должны быть нормированы)
        time (float): Время эволюции
        
    Returns:
        np.array: Вероятности для каждого уровня
    '''
    def equations(P, t):
        gamma = np.pi * F**2 / (2 * m * hbar * freq**2) * np.sin(freq * t)**2
        dPdt = np.zeros_like(P)
        
        # правила получены через представление оператора V как оператор рождения и уничтожения
        for n in range(max_level):
            # Переходы вниз |n> -> |n-1>
            if n > 0:
                dPdt[n] -= gamma * n * P[n]
                dPdt[n-1] += gamma * n * P[n]
            
            # Переходы вверх |n> -> |n+1>
            if n < max_level-1:
                dPdt[n] -= gamma * (n+1) * P[n]
                dPdt[n+1] += gamma * (n+1) * P[n]
        
        return dPdt
    
    # Решаем систему ОДУ
    solution = odeint(equations, state, [0, time])
    
    # Нормируем результат
    final_probs = solution[-1]
    final_probs /= np.sum(final_probs)
    
    return final_probs

# Пример использования
initial_state = np.zeros(max_level)
initial_state[0] = 1.0  # Начинаем с основного состояния

# Эволюция на протяжении времени t=5
final_probs = count_proba_fermi(initial_state, time)

# Визуализация результатов
import matplotlib.pyplot as plt
plt.bar(range(max_level), final_probs)
plt.xlabel('Уровень энергии n')
plt.ylabel('Вероятность P(n)')
plt.title(f'Распределение по уровням энергии при t={time}')
plt.show()
