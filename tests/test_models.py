import numpy as np
import pytest

from sklearn.linear_model import LinearRegression
from linreg.models import LinearRegressionNormal, LinearRegressionGD

TRUE_COEF = np.array([2.0, -1.0, 0.5])
TRUE_INTERCEPT = 3.0


def make_data(n=200, noise=0.0, seed=0):
    """Synthetic data where we know the right answer."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(TRUE_COEF)))
    y = X @ TRUE_COEF + TRUE_INTERCEPT + noise * rng.normal(size=n)
    return X, y


def test_normal_recovers_weights():
    X, y = make_data()
    model = LinearRegressionNormal().fit(X, y)

    assert np.allclose(model.coef_, TRUE_COEF, atol=1e-4)
    assert np.isclose(model.intercept_, TRUE_INTERCEPT, atol=1e-4)

def test_normal_matches_sklearn():
    X, y = make_data(noise=0.5)
    ours = LinearRegressionNormal().fit(X, y)
    sk = LinearRegression().fit(X, y)

    assert np.allclose(ours.coef_, sk.coef_, atol=1e-4)
    assert np.isclose(ours.intercept_, sk.intercept_, atol=1e-4)
    assert np.allclose(ours.predict(X), sk.predict(X), atol=1e-4)

@pytest.mark.parametrize(
    "method, lr, epochs",
    [
        ("batch", 0.1, 500),
        ("sgd", 0.01, 50),
        ("minibatch", 0.05, 200),
    ],
)
def test_gd_recovers_weights(method, lr, epochs):
    X, y = make_data()
    model = LinearRegressionGD(lr=lr, epochs=epochs, method=method, seed=0).fit(X, y)

    assert np.allclose(model.coef_, TRUE_COEF, atol=1e-4)
    assert np.isclose(model.intercept_, TRUE_INTERCEPT, atol=1e-4)