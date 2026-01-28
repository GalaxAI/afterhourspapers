def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.

    Args:
        a: A 2D matrix of shape (m, n)

    Returns:
        The transposed matrix of shape (n, m)
    """
    n = len(a)
    m = len(a[0])
    res = []  # mxn
    for i in range(m):
        res.append([a[j][i] for j in range(n)])

    return res
    # Also valid answer
    # [1,2,3], [4,5,6]
    # [
    # [1,4]
    # [2,5]
    # [3,6]
    # ]
    # return list(map(list, (zip(*a))))


if __name__ == "__main__":
    assert transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert transpose_matrix([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]
    assert transpose_matrix([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    assert transpose_matrix([[1, 2, 3]]) == [[1], [2], [3]]
    assert transpose_matrix([[1], [2], [3]]) == [[1, 2, 3]]
