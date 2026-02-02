import numpy as np
from tinygrad.tensor import Tensor


def linear_regression_gradient_descent_tg(X, y, alpha, iterations) -> Tensor:
    """
    Solve linear regression via gradient descent using tinygrad autograd.
    X: Tensor or convertible shape (m,n); y: shape (m,) or (m,1).
    alpha: learning rate; iterations: number of steps.
    Returns a 1-D Tensor of length n, rounded to 4 decimals.
    """
    X_t = Tensor(X).float()
    y_t = Tensor(y).float().reshape(-1, 1)
    m, n = X_t.shape
    theta = Tensor.zeros(n, 1)  # weights

    for _ in range(iterations):
        y_pred = X_t @ theta
        # loss = ((y_pred - y_t) ** 2).mean() / 2
        # mean -> 1/m
        # square gives 2* (which we cancel with /2)
        # linear gives X_t.T
        # and than our error
        grad = (1.0 / m) * (X_t.T @ (y_pred - y_t))
        theta = theta - alpha * grad
    return theta.round()


if __name__ == "__main__":
    res = linear_regression_gradient_descent_tg(X=np.array([[1, 1], [1, 2], [1, 3]]), y=np.array([3, 5, 7]), alpha=0.1, iterations=1000)
    print(res.numpy().tolist())

    res = linear_regression_gradient_descent_tg([[1.0, 0.0], [0.0, 1.0]], [5.0, 3.0], 0.1, 1000)
    print(res.numpy().tolist())

    X = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
    y = [1.0, 2.0, 3.0]
    res = linear_regression_gradient_descent_tg(X, y, 0.01, 1000)
    print(res.numpy().tolist())
