def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:

    return [[v * scalar for v in row] for row in matrix]


if __name__ == "__main__":
    assert scalar_multiply([[1, 2], [3, 4]], 2) == [[2, 4], [6, 8]]
    assert scalar_multiply([[0, -1], [1, 0]], -1) == [[0, 1], [-1, 0]]
    print("Good")
