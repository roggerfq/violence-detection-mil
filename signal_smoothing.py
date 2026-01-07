"""
Project: VD-MIL: A Deep Multiple Instance Learning Approach for Violence Detection in Surveillance Videos
Author: Roger Figueroa Quintero
Years: 2025–2026

License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

This code is part of an academic/research project.
You are free to use, modify, and share this code for non-commercial purposes only,
provided that proper credit is given to the author.

Commercial use of this code is strictly prohibited without explicit written permission
from the author.

Full license text: https://creativecommons.org/licenses/by-nc/4.0/legalcode
"""


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

    min_val = scores_smooth.min()
    max_val = scores_smooth.max()
    scores_smooth = (scores_smooth - min_val) / (max_val - min_val + 1e-8)
    
    return scores_smooth
