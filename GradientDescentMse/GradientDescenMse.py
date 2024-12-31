import pandas as pd
import numpy as np


class GradientDescentMse:
    """
    Базовый класс для реализации градиентного спуска в задаче линейной МНК регрессии
    """

    def __init__(
        self,
        samples: pd.DataFrame,
        targets: pd.DataFrame,
        learning_rate: float = 1e-3,
        threshold=1e-6,
        copy: bool = True,
    ):
        """
        self.samples - матрица признаков
        self.targets - вектор таргетов
        self.beta - вектор из изначальными весами модели == коэффициентами бета (состоит из единиц)
        self.learning_rate - параметр *learning_rate* для корректировки нормы градиента
        self.threshold - величина, меньше которой изменение в loss-функции означает остановку градиентного спуска
        iteration_loss_dict - словарь, который будет хранить номер итерации и соответствующую MSE
        copy: копирование матрицы признаков или создание изменения in-place
        """
        self.samples = samples.copy() if copy else samples
        self.targets = targets
        self.beta = np.ones(self.samples.shape[1])
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.iteration_loss_dict = {}

    def add_constant_feature(self):
        """
        Метод для создания константной фичи в матрице объектов samples.
        """
        self.samples["constant"] = 1  # Добавляем столбец с единицами
        self.beta = np.ones(self.samples.shape[1])

    def calculate_mse_loss(self) -> float:
        """
        Метод для расчета среднеквадратической ошибки

        :return: среднеквадратическая ошибка при текущих весах модели : float
        """
        error = np.dot(self.samples, self.beta) - self.targets
        return float(np.mean((error**2)))

    def calculate_gradient(self) -> np.ndarray:
        """
        Метод для вычисления вектора-градиента
        Метод возвращает вектор-градиент, содержащий производные по каждому признаку.
        Сначала матрица признаков скалярно перемножается на вектор self.beta, и из каждой колонки
        полученной матрицы вычитается вектор таргетов. Затем полученная матрица скалярно умножается на матрицу признаков.
        Наконец, итоговая матрица умножается на 2 и усредняется по каждому признаку.

        :return: вектор-градиент, т.е. массив, содержащий соответствующее количество производных по каждой переменной : np.ndarray
        """

        error = np.dot(self.samples, self.beta) - self.targets
        gradient = 2 * np.dot(error, self.samples) / self.samples.shape[0]
        return gradient

    def iteration(self):
        """
        Обновляем веса модели в соответствии с текущим вектором-градиентом
        """
        nabla_Q = self.calculate_gradient()
        self.beta = self.beta - self.learning_rate * nabla_Q

    def learn(self):
        """
        Итеративное обучение весов модели до срабатывания критерия останова
        Запись mse и номера итерации в iteration_loss_dict

        Описание алгоритма работы для изменения бет:
            Фиксируем текущие beta -> start_betas
            Делаем шаг градиентного спуска
            Записываем новые beta -> new_betas
            Пока |L(new_beta) - L(start_beta)| >= threshold:
                Повторяем первые 3 шага

        Описание алгоритма работы для изменения функции потерь:
            Фиксируем текущие mse -> previous_mse
            Делаем шаг градиентного спуска
            Записываем новые mse -> next_mse
            Пока |(previous_mse) - (next_mse)| >= threshold:
                Повторяем первые 3 шага
        """

        i = 1  # Счетчик итераций
        previous_mse = self.calculate_mse_loss()  # Текущая ошибка
        # Сохраняем начальную ошибку
        self.iteration_loss_dict[i] = previous_mse

        while True:
            self.iteration()  # Выполняем шаг градиентного спуска
            next_mse = self.calculate_mse_loss()  # Новая ошибка
            i += 1
            self.iteration_loss_dict[i] = next_mse  # Сохраняем ошибку

            # Проверяем условие остановки
            if np.abs(previous_mse - next_mse) < self.threshold:
                break

            previous_mse = next_mse  # Обновляем предыдущую ошибку
