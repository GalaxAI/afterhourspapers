def matrixmul(a: list[list[int | float]], b: list[list[int | float]]) -> list[list[int | float]]:
    if len(a[0]) != len(b):
        raise ValueError("Matrices cannot be multiplied")

    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


if __name__ == "__main__":
    assert matrixmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]
    assert matrixmul(
        [[1, 2, 3], [2, 3, 4], [5, 6, 7]],
        [[3, 2, 1], [4, 3, 2], [5, 4, 3]],
    ) == [[26, 20, 14], [38, 29, 20], [74, 56, 38]]
    assert matrixmul([[0, 0], [2, 4], [1, 2]], [[0, 0], [2, 4]]) == [[0, 0], [8, 16], [4, 8]]
    try:
        matrixmul([[0, 0], [2, 4], [1, 2]], [[0, 0, 1], [2, 4, 1], [1, 2, 3]])
    except ValueError as exc:
        assert str(exc) == "Matrices cannot be multiplied"
