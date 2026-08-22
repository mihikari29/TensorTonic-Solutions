import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape

    # Initialize parameters
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    for _ in range(steps):
        # Linear prediction
        z = X @ w + b

        # Predicted probabilities
        y_pred = _sigmoid(z)

        # Gradients of binary cross-entropy loss
        error = y_pred - y

        dw = (X.T @ error) / n_samples
        db = np.mean(error)

        # Gradient descent update
        w -= lr * dw
        b -= lr * db

    return w, b