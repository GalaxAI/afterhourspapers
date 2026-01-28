def matrix_dot_vector(a: list[list[int | float]], b: list[int | float]) -> list[int | float]:
    # Return a list where each element is the dot product of a row of 'a' with 'b'.
    # If the number of columns in 'a' does not match the length of 'b', return -1.

    if len(a[0]) != len(b):
        raise ValueError("The number of columns in 'a' does not match the length of 'b'")
    return [sum(x * y for x, y in zip(row, b)) for row in a]


if __name__ == "__main__":
    assert matrix_dot_vector([[1, 2], [2, 4]], [1, 2]) == [5, 10]
    assert matrix_dot_vector([[1, 2, 3], [2, 4, 5], [6, 8, 9]], [1, 2, 3]) == [14, 25, 49]
    try:
        matrix_dot_vector([[1, 2], [2, 4], [6, 8], [12, 4]], [1, 2, 3])
    except ValueError:
        print("Good")
    assert matrix_dot_vector([[1.5, 2.5], [3.0, 4.0]], [2, 1]) == [5.5, 10.0]
