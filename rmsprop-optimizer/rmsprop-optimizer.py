import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Perform one RMSProp update step.
    """
    # Write code here
    w = np.asarray(w, dtype = float)
    g = np.asarray(g, dtype = float)
    s = np.asarray(s, dtype = float)
    s_new = beta * s + (1 - beta) * (g ** 2)
    w_new = w - lr * g / np.sqrt(s_new + eps)
    return w_new, s_new