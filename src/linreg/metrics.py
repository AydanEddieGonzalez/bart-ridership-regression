import numpy as np

# Mean Squared Error
def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_errors = y_true - y_pred
    return (y_errors ** 2).mean()

# Root Mean Squared Error
def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

# Mean Absolute Error
def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    y_errors = y_true - y_pred
    return np.abs(y_errors).mean()

# R-Squared (coefficient of determination)
def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_true - y_pred) ** 2)          # model's squared errors
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)   # errors from just guessing the average

    # To not divide by 0
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)