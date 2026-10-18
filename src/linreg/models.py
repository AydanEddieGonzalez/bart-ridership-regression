import numpy as np

class LinearRegressionNormal():

    def __init__(self):
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        ones = np.ones(X.shape[0])
        X_bias = np.column_stack([ones, X])

        theta = np.linalg.solve(X_bias.T @ X_bias, X_bias.T @ y)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError('Call fit before predict')

        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_