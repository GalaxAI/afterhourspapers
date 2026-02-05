import numpy as np


def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations

    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros

    for _ in range(iterations):
        y_pred = X @ theta
        # loss = ((y_pred - y) ** 2).mean() / 2
        # mean -> 1/m
        # square gives 2* (which we cancel with /2)
        # linear gives X.T
        # and than our error
        grad = (1.0 / m) * (X.T @ (y_pred - y))
        theta = theta - alpha * grad
    return theta.round()


if __name__ == "__main__":
    res = linear_regression_gradient_descent(X=np.array([[1, 1], [1, 2], [1, 3]]), y=np.array([3, 5, 7]), alpha=0.1, iterations=1000)
    expected = np.array([[1.0], [2.0]])
    assert np.allclose(res, expected), f"expected {expected.tolist()} got {res.tolist()}"
    print(res.tolist())

    res = linear_regression_gradient_descent(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([5.0, 3.0]), 0.1, 1000)
    expected = np.array([[5.0], [3.0]])
    assert np.allclose(res, expected), f"expected {expected.tolist()} got {res.tolist()}"
    print(res.tolist())

    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 3.0])
    res = linear_regression_gradient_descent(X, y, 0.01, 1000)
    expected = np.array([[0.0], [1.0]])
    assert np.allclose(res, expected), f"expected {expected.tolist()} got {res.tolist()}"
    print(res.tolist())
