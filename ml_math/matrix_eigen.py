import operator
from functools import reduce


def determinant(matrix: list[list[float | int]]) -> float:
    n = len(matrix)
    if n != len(matrix[0]):
        raise ValueError("Matrix must be square")
    if n == 1:
        return float(matrix[0][0])
    if n == 2:
        return float(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])

    det = 0.0
    # Sarrus' rule for 3x3 matrices
    for j in range(n):
        positive_diagonal = [row[(j + idx) % n] for idx, row in enumerate(matrix)]
        negative_diagonal = [row[-(j + idx) % n] for idx, row in enumerate(matrix)]
        det += reduce(operator.mul, positive_diagonal)
        det -= reduce(operator.mul, negative_diagonal)
    return det


def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    if len(matrix) != 2 or len(matrix[0]) != 2:
        raise ValueError("Only 2x2 matrices supported")

    # Extract elements
    a, b = matrix[0]
    c, d = matrix[1]

    # Characteristic equation: λ² - trace*λ + det = 0
    trace = a + d
    det = determinant(matrix)

    # Quadratic formula: λ = (trace ± sqrt(trace² - 4*det)) / 2
    discriminant = trace**2 - 4 * det

    if discriminant < 0:
        raise ValueError("Eigenvalues are complex")

    eigenvalue1 = (trace + (discriminant) ** 0.5) / 2
    eigenvalue2 = (trace - (discriminant) ** 0.5) / 2

    return [eigenvalue1, eigenvalue2]


if __name__ == "__main__":
    assert calculate_eigenvalues([[2, 1], [1, 2]]) == [3.0, 1.0]
    assert calculate_eigenvalues([[4, -2], [1, 1]]) == [3.0, 2.0]
