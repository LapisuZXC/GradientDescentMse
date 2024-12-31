import pandas as pd
import numpy as np


class GradientDescentMse:
    """
    Base class for implementing gradient descent in linear MSE regression
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
        Initialize parameters:
        - samples: feature matrix
        - targets: target vector
        - beta: model weights (initialized to ones)
        - learning_rate: learning rate for gradient correction
        - threshold: convergence threshold
        - iteration_loss_dict: dictionary to store iteration number and MSE
        - copy: whether to copy the feature matrix or modify it in place
        """
        self.samples = samples.copy() if copy else samples
        self.targets = targets
        self.beta = np.ones(self.samples.shape[1])
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.iteration_loss_dict = {}

    def add_constant_feature(self):
        """
        Adds a constant feature to the feature matrix.
        """
        self.samples["constant"] = 1  # Add a column of ones
        self.beta = np.ones(self.samples.shape[1])

    def calculate_mse_loss(self) -> float:
        """
        Calculate the mean squared error (MSE).
        :return: MSE for the current model weights
        """
        error = np.dot(self.samples, self.beta) - self.targets
        return float(np.mean((error**2)))

    def calculate_gradient(self) -> np.ndarray:
        """
        Compute the gradient vector.
        :return: gradient vector with partial derivatives for each feature
        """
        error = np.dot(self.samples, self.beta) - self.targets
        gradient = 2 * np.dot(error, self.samples) / self.samples.shape[0]
        return gradient

    def iteration(self):
        """
        Update model weights using the current gradient.
        """
        nabla_Q = self.calculate_gradient()
        self.beta = self.beta - self.learning_rate * nabla_Q

    def learn(self):
        """
        Iteratively train model weights until the convergence criterion is met.
        MSE and iteration number are recorded in iteration_loss_dict.

        Algorithm for beta updates:
        - Fix the current beta -> start_betas
        - Perform a gradient descent step
        - Record new beta -> new_betas
        - Repeat until |L(new_beta) - L(start_beta)| < threshold

        Algorithm for loss function updates:
        - Fix the current MSE -> previous_mse
        - Perform a gradient descent step
        - Record new MSE -> next_mse
        - Repeat until |previous_mse - next_mse| < threshold
        """
        i = 1  # Iteration counter
        previous_mse = self.calculate_mse_loss()
        self.iteration_loss_dict[i] = previous_mse

        while True:
            self.iteration()
            next_mse = self.calculate_mse_loss()
            i += 1
            self.iteration_loss_dict[i] = next_mse

            # Check the convergence condition
            if np.abs(previous_mse - next_mse) < self.threshold:
                break

            previous_mse = next_mse
