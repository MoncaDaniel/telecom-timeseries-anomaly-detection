# src/features/windowing.py
import numpy as np

def make_windows(arr_2d: np.ndarray, window_size: int) -> np.ndarray:
    """
    Convert a 2D array (T, F) into windows (N, window_size, F)
    where each window i covers arr[i : i+window_size].

    T = number of time points
    F = number of features (KPIs)
    """
    T = arr_2d.shape[0]
    if T <= window_size:
        return np.empty((0, window_size, arr_2d.shape[1]), dtype=np.float32)

    windows = [arr_2d[i:i+window_size] for i in range(T - window_size)]
    return np.asarray(windows, dtype=np.float32)
