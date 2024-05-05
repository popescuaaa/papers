import numpy as np
from typing import Tuple, List


class SimpleLinearRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        # Data
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None

        # OG data
        self.x = x  # points
        self.y = y  # labels, but they are actually values

        # Params
        self.weight = None
        self.bias = None

    @staticmethod
    def mse(real: np.ndarray, predicted: np.ndarray) -> float:
        return np.mean((real - predicted) ** 2)

    @staticmethod
    def split(x: np.ndarray, y: np.ndarray, fraction: float = 0.2) -> (
            Tuple)[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_train = x.copy()[:int(len(x) * fraction)]
        y_train = y.copy()[:int(len(x) * fraction)]

        x_test = x.copy()[int(len(x) * fraction):]
        y_test = y.copy()[int(len(x) * fraction):]

        return x_train, y_train, x_test, y_test

    def normalize(self) -> None:
        max_value = np.max(self.x)
        min_value = np.min(self.x)
        self.x = (self.x - min_value) / (max_value - min_value)

    def loss(self, step: str = "train") -> float:
        """"
        Loss function: MSE
        """
        if step == "train":
            N = self.x_train.shape[0]
            total_loss = 0.0
            for i in range(N):
                total_loss += self.mse(real=self.y_train[i], predicted=(self.weight * self.x_train[i] + self.bias))

            return total_loss

        elif step == "test":
            N = self.x_test.shape[0]
            total_loss = 0.0
            for i in range(N):
                total_loss += self.mse(real=self.y_test[i], predicted=(self.weight * self.x_test[i] + self.bias))

            return total_loss

        else:
            raise ValueError(f"Unrecognized step: {step}")

    def fit(self) -> None:
        self.weight = 1
        self.bias = 0
        self.x_train, self.y_train, self.x_test, self.y_test = self.split(x=self.x, y=self.y, fraction=0.2)

    def compute_gradients(self, learning_rate: float) -> None:
        """
        The loss function is:

         loss(weight, bias) = MSE(y, (weight * x_train + bias)) = || y - (weight * x_train + bias) || ** 2

         for future updates we need:
          - dloss/ dw (weight)
          - dloss/ db (bias)

         Then we make the updates using the gradient descent rule:
         w* = w - dloss/ dw (weight) * alpha
         b* = b - dloss/ db (bias) * alpha

        where alpha is the learning rate.
        """

        delta_weight = 0
        delta_bias = 0
        num_samples = len(self.x_train)  # don't use batches

        for i in range(num_samples):
            _delta_weight = -2 * self.x_train[i] * (self.y_train[i] - (self.weight * self.x_train[i] + self.bias))
            _delta_bias = -2 * (self.y_train[i] - (self.weight * self.x_train[i] + self.bias))

            delta_weight = float(_delta_weight)
            delta_bias = float(_delta_bias)

        self.weight -= (delta_weight / num_samples) * learning_rate
        self.bias -= (delta_bias / num_samples) * learning_rate

    def train(self, iterations: int, learning_rate: float) -> List[float]:
        loss_history = []
        for i in range(iterations):
            # compute gradient
            self.compute_gradients(learning_rate=learning_rate)
            loss = self.loss(step="train")
            loss_history.append(loss)

        return loss_history

    def predict(self, x: np.ndarray) -> np.ndarray:
        num_samples = x.shape[0]
        predictions = np.zeros(num_samples)
        for i in range(num_samples):
            predictions[i] = (self.weight * x[i] + self.bias)

        return predictions
