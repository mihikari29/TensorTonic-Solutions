def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = x0
    for _ in range(steps):
        error = 2 * a * x + b
        dx = lr * error
        x = x - dx
    return x