import numpy as np


def calculate_dot_product(vec1: np.ndarray, vec2: np.ndarray) -> int | float:
    """
    Calculate the dot product of two vectors.
    Args:
        vec1 (numpy.ndarray): 1D array representing the first vector.
        vec2 (numpy.ndarray): 1D array representing the second vector.
    Returns:
        The dot product of the two vectors.
    """
    return sum([x * y for x, y in zip(vec1, vec2)])
    # return vec1.dot(vec2) This is cheating


if __name__ == "__main__":
    assert calculate_dot_product(np.array([1, 2, 3]), np.array([4, 5, 6])) == 32
    assert calculate_dot_product(np.array([-1, 2, 3]), np.array([4, -5, 6])) == 4
    assert calculate_dot_product(np.array([1, 0]), np.array([0, 1])) == 0
    assert calculate_dot_product(np.array([0, 0, 0]), np.array([0, 0, 0])) == 0
    assert calculate_dot_product(np.array([7]), np.array([3])) == 21
    print("Good")
