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


class LinearRegressionGD:

    def __init__(self, lr=0.01, epochs=100, method='batch', batch_size=32, seed=None):
        self.lr = lr
        self.epochs = epochs
        self.method = method
        self.batch_size = batch_size
        self.seed = seed

        self.intercept_ = None
        self.coef_ = None
        self.loss_history_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        ones = np.ones(X.shape[0])
        X_bias = np.column_stack([ones, X])

        n = X_bias.shape[0]
        theta = np.zeros(X_bias.shape[1])

        if self.method == 'batch':
            size = n
        elif self.method == 'sgd':
            size = 1
        elif self.method == 'minibatch':
            size = self.batch_size
        else:
            raise ValueError('Assign a correct batch type')

        rng = np.random.default_rng(self.seed)
        self.loss_history_ = []

        for epoch in range(self.epochs):
            order = rng.permutation(n)
            X_shuf = X_bias[order]
            y_shuf = y[order]

            for start in range(0, n, size):
                X_chunk = X_shuf[start:start + size]
                y_chunk = y_shuf[start:start + size]

                errors = X_chunk @ theta - y_chunk
                gradient = X_chunk.T @ errors / len(y_chunk)
                theta = theta - self.lr * gradient


            full_errors = X_bias @ theta - y
            loss = (full_errors ** 2).mean() / 2
            self.loss_history_.append(loss)

            if not np.isfinite(loss):
                break

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self
            

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError('Call fit before predict')

        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_