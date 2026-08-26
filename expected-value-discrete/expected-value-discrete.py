import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    """
    Returns the expected value as a Python float.
    """
    # Write code here
    res = float(0)
    for i in range(len(x)):
        res += x[i] * p[i]
    return res