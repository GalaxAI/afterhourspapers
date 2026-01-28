import numpy as np

from .dot_product import calculate_dot_product


def l2_norm(v1: np.ndarray) -> float | int:
    """
    Calculates L2 norm of the vectors
    Arg:
        vec1 (numpy.ndarray): 1D array representing the vector
    Returns:
        The l2 norm of the vector
    """
    return ((v1**2).sum()) ** 0.5


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float | int:
    """
    Calculate the cosine_similarity of two vectors.
    Args:
        vec1 (numpy.ndarray): 1D array representing the first vector.
        vec2 (numpy.ndarray): 1D array representing the second vector.
    Returns:
        The cosine_similarity of the two vectors.
    """
    top = calculate_dot_product(v1, v2)
    v1_l2 = l2_norm(v1)
    v2_l2 = l2_norm(v2)
    return top / (v1_l2 * v2_l2)


if __name__ == "__main__":
    """
    # Note: run this file with
    uv run -m ml_math.cos_similarity
    """
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 4, 6])
    assert round(cosine_similarity(v1, v2), 3) == 1.0
    v1, v2 = np.array([1, 2, 3]), np.array([-1, -2, -3])
    assert round(cosine_similarity(v1, v2), 3) == -1.0
    v1 = np.array([1, 0, 7])
    v2 = np.array([0, 1, 3])
    assert round(cosine_similarity(v1, v2), 3) == 0.939
    print("Good")
