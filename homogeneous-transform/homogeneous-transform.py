import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transform to one point or a batch.
    """
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    if points.ndim == 1:
        p_h = np.append(points, 1.0)
        transformed = T @ p_h
        return transformed[:3]

    ones = np.ones((points.shape[0], 1))
    p_h = np.hstack((points, ones))

    transformed = (T @ p_h.T).T
    return transformed[:, :3]