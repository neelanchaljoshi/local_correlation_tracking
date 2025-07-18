import numpy as np
def sin_fit(t, a, b, c, d, e):
    """
    Fit a sinusoidal function to the given time data.
    Parameters:
        t (numpy.ndarray): Time data in years.
        a (float): Amplitude of the first sine component.
        b (float): Amplitude of the first cosine component.
        c (float): Amplitude of the second sine component.
        d (float): Amplitude of the second cosine component.
        e (float): Offset value.
    Returns:
        numpy.ndarray: The fitted sinusoidal function evaluated at time t.
    """
    return a * np.sin(2*np.pi*t) + b * np.cos(2*np.pi*t) + c * np.sin(4*np.pi*t) + d * np.cos(4*np.pi*t) + e