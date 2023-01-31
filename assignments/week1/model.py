import numpy as np


class LinearRegression:
    """
    linear regression using Normal Equation
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        fit the model with following inputs
        """
        cols = X.shape[1]
        X = np.append(X,np.ones((X.shape[0],1)),axis=1)
        y = y.reshape(-1, 1)
        self.w = (np.linalg.inv(X.T @ X)) @ X.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict.
        """
        cols = X.shape[1]
        X = X.reshape(-1, cols)
        return np.dot(X, self.w) + self.b


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        fit the model with following inputs
        """
        self.w = 0
        self.b = 0
        size = X.shape[0]
        for i in range(epochs):
            y_pred = self.w * X + self.b
            dw = (-2 / size) * sum(X * (y - y_pred))
            db = (-1 / size) * sum(y - y_pred)
            self.w -= lr * dw
            self.b -= lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return np.dot(X, self.w) + self.b
