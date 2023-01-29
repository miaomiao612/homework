
import numpy as np


class LinearRegression:
    """
    linear regression using Normal Equation
    """
    w: np.ndarray
    b: float

    def __init__(self):
        self.w=0
        self.b=0
    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        fit the model with following inputs
        """
        cols=X.shape[1]
        X=np.array(X).reshape(-1,cols)
        y=np.array(y).reshape(-1,1)
        self.w = np.dot((np.linalg.inv(np.dot(X.T, X))), np.dot(X.T, y))

    def predict(self, X):
        """
        predict
        """
        X=np.array(X).reshape(-1,1)
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
        size = X.shape[0]
        for _ in range(epochs):
            y_pred = np.dot(X, self.w) + self.b
            dw = (2 / size) * np.dot(X.T, (y_pred - y))
            db = (1 / size) * np.sum(y_pred - y)
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

