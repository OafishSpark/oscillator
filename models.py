import numpy as np

from math import factorial
from scipy.special import hermite
from scipy.integrate import simpson


def stationary_state_wavefunction(n, x):
    """
    Волновая функция стационарных состояний от обощенных нормированных координат (без времени).
    
    Параметр:
    n: int
        Число фотонов
    x: np.array
        Массив с координатами
    
    Возвращает:
        Массив с амплитудами вероятности для каждой координаты 
    
    """
    # Нормировочный множитель
    norm = 1.0 / np.sqrt(2**n * factorial(n)) * (1.0 / (np.pi))**0.25
    # Полином Эрмита
    Hn = hermite(n)
    xi = x
    # Волновая функция
    psi = norm * Hn(xi) * np.exp(-xi**2 / 2)
    return psi


class HarmonicOscillator:
    def __init__(self, frequency, 
                 current_time = 0.0,
                 x_grid = np.linspace(-5, 5, 1000),
                 wave_function = None,
                 num_states = 10,
                 coefficients = None
    ):
        """
        Конструктор класса одномерного гармонического осциллятора.
        
        Параметры:
        frequency : float
            Частота колебаний
        current_time : float
            Текущий момент времени
        x_grid : numpy.ndarray
            Сетка по координате
        wave_function : list
            Значение волновой функции в координате
        num_states : int
            Число стационарных состояний по которым раскладываем
        coefficients : list
            Коэффициенты при разложении по стационарным состояниям

        """
        self.frequency = frequency
        self.current_time = current_time
        self.x_grid = x_grid
        self.num_states = num_states
        if coefficients == None:
          self.wave_function = wave_function
          self.compute_states_coefficients(num_states)
        else:
          self.coefficients = coefficients
          self.reconstruct_wavefunction()
        

    def compute_states_coefficients(self, max_n=None):
        """
        Вычисляет коэффициенты разложения текущей волновой функции
        по стационарным состояниям гармонического осциллятора.
        
        Параметры:
        max_n : int или None
            Максимальный номер состояния для вычисления.
            Если None, используется self.num_states
        
        Возвращает:
        coefficients : dict
            Словарь {n: c_n} коэффициентов разложения
        """
        if max_n is None:
            max_n = self.num_states
        
        coefficients = np.zeros(max_n, dtype=complex)

        for n in range(max_n):
            # Вычисляем волновую функцию n-го стационарного состояния
            psi_n = stationary_state_wavefunction(n, self.x_grid)
            
            # Вычисляем коэффициент
            temp = simpson(np.conj(self.wave_function) * psi_n, self.x_grid)
            coefficients[n] = temp * np.exp(1j * self.frequency * (n + 1/2) * self.current_time)
            
        # Обновляем внутренние параметры класса
        self.coefficients = coefficients
        self.num_states = max_n
        
        return coefficients


    def reconstruct_wavefunction(self, x_grid=None, max_n=None):
        """
        Восстанавливает волновую функцию по коэффициентам разложения
        
        Параметры:
        x_grid : numpy.ndarray или None
            Сетка по координате (если None, используется self.x_grid)
        max_n : int или None
            Максимальное число учитываемых состояний (если None, используются все)
      
        Возвращает:
        dict {координата: значение волновой функции}
        """
        if x_grid is None:
            x_grid = self.x_grid
            
        if max_n is None:
            max_n = self.num_states
        
        # Инициализируем волновую функцию нулями
        psi = np.zeros_like(x_grid, dtype=complex)
        
        # Суммируем вклады всех состояний
        for n, cn in enumerate(self.coefficients):
            # Получаем волновую функцию n-го состояния
            psi_n = stationary_state_wavefunction(n, x_grid)
            # Добавляем вклад этого состояния
            psi += cn * psi_n
        
        # Обновляем состояние класса
        self.wave_function = psi
        
        return psi


    def fock_evolution(self, t):
        self.current_time += t
        # Инициализируем волновую функцию нулями
        psi = np.zeros_like(self.x_grid, dtype=complex)
          
        # Суммируем вклады всех состояний
        for n, cn in enumerate(self.coefficients):
            # Получаем волновую функцию n-го состояния
            psi_n = stationary_state_wavefunction(n, self.x_grid)
            # Добавляем вклад этого состояния
            psi += cn * psi_n * np.exp(-1j * self.frequency * (n + 1/2) * self.current_time)
            self.coefficients[n] = cn * np.exp(-1j * self.frequency * (n + 1/2) * self.current_time)
        
        # Обновляем состояние класса
        self.wave_function = psi
        
        return psi


    def compute_coherent_coefficient(self, alpha):
        """
        Вычисляет коэффициент разложения текущей волновой функции по когерентному состоянию alpha. (phi_0 = 0)
        
        Параметры:
        alpha : complex
            Параметр когерентного состояния.
        
        Возвращает:
        complex
            Коэффициент C_alpha
        """
        
        c_alpha = 0
        for n, coeff in enumerate(self.coefficients):
            c_alpha += coeff * alpha**n / np.sqrt(factorial(n)) * self.coefficients[n]
        c_alpha *= np.exp(-abs(alpha)**2/2)
        return c_alpha


    def compute_average_values(self):
            """
            Вычисляет средние значения координаты, импульса и энергии.
            
            Возвращает:
            tuple (avg_x, avg_p, avg_E)
                Средние значения координаты, импульса и энергии
            """
            psi = self.wave_function
            x = self.x_grid
            dx = x[1] - x[0]
            
            # Вычисляем производную волновой функции для импульса
            dpsi_dx = np.gradient(psi, dx)
            
            # Средняя координата
            avg_x = simpson(np.conj(psi) * x * psi, x)
            
            # Средний импульс (в единицах hbar=1)
            avg_p = -1j * simpson(np.conj(psi) * dpsi_dx, x)
            
            # Средняя энергия через коэффициенты разложения
            avg_E = 0
            for n, cn in enumerate(self.coefficients):
                avg_E += np.abs(cn)**2 * (n + 0.5) * self.frequency
                
            return avg_x.real, avg_p.real, avg_E.real
        
    def compute_standard_deviations(self):
            """
            Вычисляет среднеквадратичные отклонения координаты, импульса и энергии.
            
            Возвращает:
            tuple (sigma_x, sigma_p, sigma_E)
                Среднеквадратичные отклонения координаты, импульса и энергии
            """
            psi = self.wave_function
            x = self.x_grid
            dx = x[1] - x[0]
            
            # Вычисляем производные волновой функции
            dpsi_dx = np.gradient(psi, dx)
            
            # Средние значения
            avg_x, avg_p, avg_E = self.compute_average_values()
            
            # Средний квадрат координаты
            avg_x2 = simpson(np.conj(psi) * x**2 * psi, x)
            sigma_x = np.sqrt(avg_x2.real - avg_x**2)
            
            # Средний квадрат импульса
            avg_p2 = -simpson(np.conj(psi) * np.gradient(dpsi_dx, dx), x)
            sigma_p = np.sqrt(avg_p2.real - avg_p**2)
            
            # Среднеквадратичное отклонение энергии
            avg_E2 = 0
            for n, cn in enumerate(self.coefficients):
                avg_E2 += np.abs(cn)**2 * ((n + 0.5) * self.frequency)**2
            sigma_E = np.sqrt(avg_E2.real - avg_E**2)
            
            return sigma_x, sigma_p, sigma_E
        
    def average_values_dynamics(self, time_interval, num_points=100):
            """
            Выводит динамику средних значений координаты, импульса и энергии
            на заданном временном интервале.
            
            Параметры:
            time_interval : tuple (t_start, t_end)
                Начальный и конечный моменты времени
            num_points : int
                Число точек на интервале
                
            Возвращает:
            tuple (times, avg_x, avg_p, avg_E)
                Массивы времен, средних координат, импульсов и энергий
            """
            t_start, t_end = time_interval
            times = np.linspace(t_start, t_end, num_points)
            
            avg_x = np.zeros(num_points, dtype=float)
            avg_p = np.zeros(num_points, dtype=float)
            avg_E = np.zeros(num_points, dtype=float)
            
            # Сохраняем исходное состояние
            original_time = self.current_time
            original_coefficients = self.coefficients.copy()
            
            for i, t in enumerate(times):
                self.fock_evolution(times[1]-times[0])  # Обновляем волновую функцию для времени t
                avg_x[i], avg_p[i], avg_E[i] = self.compute_average_values()
            
            # Восстанавливаем исходное состояние
            self.current_time = original_time
            self.coefficients = original_coefficients
            self.reconstruct_wavefunction()
            
            return times, avg_x, avg_p, avg_E
        
    def standard_deviations_dynamics(self, time_interval, num_points=100):
            """
            Выводит динамику среднеквадратичных отклонений координаты, импульса и энергии
            на заданном временном интервале.
            
            Параметры:
            time_interval : tuple (t_start, t_end)
                Начальный и конечный моменты времени
            num_points : int
                Число точек на интервале
                
            Возвращает:
            tuple (times, sigma_x, sigma_p, sigma_E)
                Массивы времен, СКО координат, импульсов и энергий
            """
            t_start, t_end = time_interval
            times = np.linspace(t_start, t_end, num_points)
            
            sigma_x = np.zeros(num_points, dtype=float)
            sigma_p = np.zeros(num_points, dtype=float)
            sigma_E = np.zeros(num_points, dtype=float)
            
            # Сохраняем исходное состояние
            original_time = self.current_time
            original_coefficients = self.coefficients.copy()
            
            for i, t in enumerate(times):
                self.fock_evolution(times[1]-times[0])  # Обновляем волновую функцию для времени t
                sigma_x[i], sigma_p[i], sigma_E[i] = self.compute_standard_deviations()
            
            # Восстанавливаем исходное состояние
            self.current_time = original_time
            self.coefficients = original_coefficients
            self.reconstruct_wavefunction()
            
            return times, sigma_x, sigma_p, sigma_E
