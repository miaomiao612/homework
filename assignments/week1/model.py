'''
Author: miaomiao612 dddoctorr612@gmail.com
Date: 2023-01-23 04:30:43
LastEditors: miaomiao612 dddoctorr612@gmail.com
LastEditTime: 2023-01-28 09:38:37
FilePath: \week1\model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np


class LinearRegression:

    w: np.ndarray
    b: float

    def __init__(self):
        raise NotImplementedError()

    def fit(self, X, y):
        #raise NotImplementedError()
        self.X=X
        self.y=y
        n_features = X.shape
        self.w = np.zeros(shape=(n_features, 1))
        self.b = 0.0

    def predict(self, X):
        #raise NotImplementedError()
        y_pred = np.dot(X,self.w)+self.b
        return y_pred

class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        #raise NotImplementedError()
        n_samples, n_features = X.shape
        self.w = np.zeros(shape=(n_features, 1))
        self.b = 0.0
        for _ in range(epochs):
            y_pred = np.dot(X, self.w) + self.b
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
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
        #raise NotImplementedError()






