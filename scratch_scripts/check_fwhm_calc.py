# %%
import numpy as np
def fwhm_from_xy(x, y, baseline=None, x_sorted=False):
    """
    Compute FWHM (full width at half maximum) of a single-peaked, Lorentzian-like curve.

    Parameters
    ----------
    x : array-like
        Independent variable values (x-axis). Can be unsorted.
    y : array-like
        Dependent variable values (y-axis).
    baseline : float or None, optional
        Baseline (offset) to subtract from y before computing half-maximum.
        If None, baseline = min(y).
    x_sorted : bool
        If True, assumes x is already strictly increasing. If False, function will sort x,y.

    Returns
    -------
    fwhm : float or np.nan
        Full width at half maximum in units of x (x_right - x_left). np.nan if not computable.
    x_left : float or None
        x position of left half-maximum crossing (interpolated).
    x_right : float or None
        x position of right half-maximum crossing (interpolated).
    peak_x : float
        x position of the peak.
    peak_y : float
        y value of the peak (after baseline subtraction).

    Notes
    -----
    - Works for a single peak (one maximum). If the data contain multiple peaks, supply sliced data around the peak.
    - Uses linear interpolation between samples to estimate crossing points.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # sort by x unless told sorted
    if not x_sorted:
        order = np.argsort(x)
        x = x[order]
        y = y[order]

    # baseline
    if baseline is None:
        baseline = np.min(y)

    y0 = y - baseline
    # avoid negative or zero plateau: if entire y0 <= 0, cannot compute
    if np.all(y0 <= 0):
        return np.nan, None, None, None, None

    # find peak
    peak_idx = np.argmax(y0)
    peak_y = y0[peak_idx]
    peak_x = x[peak_idx]

    half = peak_y / 2.0

    # left side: search indices i where y0[i] <= half < y0[i+1] (i runs 0..peak_idx-1)
    x_left = None
    if peak_idx == 0:
        x_left = None
    else:
        left_slice_y = y0[:peak_idx+1]
        left_slice_x = x[:peak_idx+1]
        # find last index i before peak where y0[i] <= half
        # we look for i such that left_slice_y[i] <= half <= left_slice_y[i+1]
        idxs = np.where(left_slice_y <= half)[0]
        if idxs.size == 0:
            # maybe the values on left are all above half (rare) -> cannot find crossing
            x_left = None
        else:
            i = idxs[-1]
            # handle boundary: if i == peak_idx then no crossing found
            if i == peak_idx:
                x_left = None
            else:
                x1, y1 = left_slice_x[i], left_slice_y[i]
                x2, y2 = left_slice_x[i+1], left_slice_y[i+1]
                if y2 == y1:
                    x_left = x1
                else:
                    t = (half - y1) / (y2 - y1)
                    x_left = x1 + t * (x2 - x1)

    # right side: search indices i where y0[i] >= half > y0[i+1] (i runs peak_idx..end-2)
    x_right = None
    if peak_idx == len(x) - 1:
        x_right = None
    else:
        right_slice_y = y0[peak_idx:]
        right_slice_x = x[peak_idx:]
        # find first index j (relative to peak_idx) where right_slice_y[j] <= half
        idxs = np.where(right_slice_y <= half)[0]
        if idxs.size == 0:
            x_right = None
        else:
            j = idxs[0]
            # if j == 0, the first point at peak is already <= half -> can't interpolate to right
            if j == 0:
                x_right = None
            else:
                i = peak_idx + (j - 1)
                x1, y1 = x[i], y0[i]
                x2, y2 = x[i+1], y0[i+1]
                if y2 == y1:
                    x_right = x2
                else:
                    t = (half - y1) / (y2 - y1)
                    x_right = x1 + t * (x2 - x1)

    if (x_left is None) or (x_right is None):
        return np.nan, x_left, x_right, peak_x, peak_y

    fwhm = x_right - x_left
    return fwhm, x_left, x_right, peak_x, peak_y
# %%
# Generate a lorentzian-like curve and test the function
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 100)
# y without randomness
y = 1 / (1 + (x / 2)**2)
fwhm, x_left, x_right, peak_x, peak_y = fwhm_from_xy(x, y)
print(f"FWHM: {fwhm}, x_left: {x_left}, x_right: {x_right}, peak_x: {peak_x}, peak_y: {peak_y}")
plt.plot(x, y, label='Data')
plt.axhline(peak_y / 2, color='red', linestyle='--', label='Half Maximum')
if x_left is not None:
    plt.axvline(x_left, color='green', linestyle='--', label='FWHM Left')
if x_right is not None:
    plt.axvline(x_right, color='orange', linestyle='--', label='FWHM Right')
plt.scatter([peak_x], [peak_y], color='purple', label='Peak')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('FWHM Calculation Test')
plt.show()

# %%
