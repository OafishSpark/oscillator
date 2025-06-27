from models import HarmonicOscillator, stationary_state_wavefunction

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson


def show_graphs():
    frequency = 1
    x_grid = np.linspace(-10, 10, 10000)
    num_steps = 100

    oscillator = HarmonicOscillator(
        frequency=1,
        x_grid=x_grid,
        current_time=0,
        coefficients=[1/np.sqrt(2), 1/np.sqrt(2)]
    )

    period = 2*np.pi / frequency
    time_interval = (0, period)
    times, avg_x, avg_p, avg_E = oscillator.average_values_dynamics(time_interval, num_steps)
    times, sigma_x, sigma_p, sigma_E = oscillator.standard_deviations_dynamics(time_interval, num_steps)

    # print(avg_p)
    # print(avg_x)
    # print(avg_E)

    # Графики средних значений
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(times, avg_x)
    plt.title('Средняя координата')
    plt.xlabel('Время')
    plt.ylabel('<x>')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(times, avg_p)
    plt.title('Средний импульс')
    plt.xlabel('Время')
    plt.ylabel('<p>')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(times, avg_E)
    plt.title('Средняя энергия')
    plt.xlabel('Время')
    plt.ylabel('<E>')
    plt.grid(True)

    plt.tight_layout()
    plt.suptitle('Динамика средних значений', y=1.02)
    # plt.show()

    # Графики среднеквадратичных отклонений
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(times, sigma_x)
    plt.title('СКО координаты')
    plt.xlabel('Время')
    plt.ylabel('σ_x')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(times, sigma_p)
    plt.title('СКО импульса')
    plt.xlabel('Время')
    plt.ylabel('σ_p')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(times, sigma_E)
    plt.title('СКО энергии')
    plt.xlabel('Время')
    plt.ylabel('σ_E')
    plt.grid(True)

    plt.tight_layout()
    plt.suptitle('Динамика среднеквадратичных отклонений', y=1.02)
    # plt.show()

    # Соотношение неопределенностей
    uncertainty = sigma_x * sigma_p
    plt.figure(figsize=(8, 5))
    plt.plot(times, uncertainty)
    plt.axhline(0.5, color='r', linestyle='--', label='Граница неопределенности')
    plt.title('Произведение σ_x·σ_p')
    plt.xlabel('Время')
    plt.ylabel('σ_x·σ_p')
    plt.legend()
    plt.grid(True)
    plt.show()

def correctness():
    x_grid = np.linspace(-10, 10, 10000)
    oscillator_1 = HarmonicOscillator(
        frequency=1,
        x_grid=x_grid,
        current_time=0,
        wave_function=np.array([np.sqrt(np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)) for x in x_grid]),
    )
    print(oscillator_1.coefficients[0])

    oscillator_2 = HarmonicOscillator(
        frequency=1,
        x_grid=x_grid,
        current_time=0,
        coefficients=[1]
    )
    # Где-то в нуле проверяем
    print(abs(oscillator_2.wave_function[len(oscillator_2.wave_function) // 2]) ** 2)
    # Проверка нормировки
    simpson(abs(oscillator_2.wave_function)**2, x_grid)
    # Сделаем один оборот
    oscillator_2.fock_evolution(4*np.pi / oscillator_2.frequency)
    print(oscillator_2.coefficients)



if __name__ == "__main__":
    
    # correctness()
    
    # Вычисляем коэффициент разложения по когерентному состоянию
    oscillator_2 = HarmonicOscillator(
        frequency=1,
        x_grid=np.linspace(-10, 10, 10000),
        current_time=0,
        coefficients=[1]
    )
    alpha = 1.0 + 0.0j
    c_alpha = oscillator_2.compute_coherent_coefficient(alpha)
    print(f"Коэффициент разложения по когерентному состоянию {alpha}: {c_alpha}")
    
    show_graphs()