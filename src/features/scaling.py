# src/features/scaling.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fit_global_minmax_scaler(list_of_arrays_2d):
    """
    Fit a single MinMaxScaler on concatenated NORMAL data.
    Each array is shape (T, F). We stack on time to get (sum_T, F).
    """
    stacked = np.vstack(list_of_arrays_2d).astype(np.float32)
    scaler = MinMaxScaler()
    scaler.fit(stacked)
    return scaler

def transform_with_scaler(scaler, arr_2d):
    return scaler.transform(arr_2d.astype(np.float32))
