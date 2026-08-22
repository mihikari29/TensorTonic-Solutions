import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Vectorized sigmoid function.
    """
    # Write code here
    return 1 / (1 + np.exp(-np.asarray(x)))