from typing import Literal

Mode = Literal["column", "row"]


def calculate_matrix_mean(matrix: list[list[float | int]], mode: Mode) -> list[float]:
    """Return per-row or per-column means for a rectangular matrix.

    Args:
        a: Rectangular matrix of numbers.
        mode: "row" for row means, "column" for column means.
    """
    match mode:
        case "row":
            return [sum(row) / len(row) for row in matrix]
        case "column":
            n_rows = len(matrix)
            n_cols = len(matrix[0])
            return [sum(matrix[r][c] for r in range(n_rows)) / n_rows for c in range(n_cols)]
        case _:
            raise ValueError("Missing value for mode")


if __name__ == "__main__":
    assert calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "column") == [4.0, 5.0, 6.0]
    assert calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], "row") == [2.0, 5.0, 8.0]
    assert calculate_matrix_mean([[1, 2, 3, 4], [5, 6, 7, 8]], "row") == [2.5, 6.5]
    assert calculate_matrix_mean([[1, 2, 3, 4], [5, 6, 7, 8]], "column") == [3.0, 4.0, 5.0, 6.0]
    print("Good")
