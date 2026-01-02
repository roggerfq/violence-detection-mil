import numpy as np
from scipy.signal import savgol_filter

def smooth_scores(scores, polyorder=3, window_ratio=0.05):
    """
    Smooth a 1D list of scores using Savitzky–Golay filtering.

    Parameters:
        scores (list or array): Input signal to smooth.
        polyorder (int): Polynomial order of the filter.
        window_ratio (float): Fraction of signal length used for window size.

    Returns:
        np.ndarray: Smoothed signal with same length as input.
    """
    y = np.asarray(scores)
    L = len(y)

    # Base window estimation from signal length
    window_length = max(int(window_ratio * L), polyorder + 2)

    # Ensure window length is odd
    if window_length % 2 == 0:
        window_length += 1

    # Ensure window length is smaller than signal length
    if window_length >= L:
        window_length = L - 1 if (L - 1) % 2 == 1 else L - 2

    # Ensure polynomial order is strictly smaller than window length
    if polyorder >= window_length:
        polyorder = window_length - 1

    # Apply Savitzky–Golay filter
    scores_smooth = savgol_filter(y, window_length, polyorder)

    return scores_smooth
