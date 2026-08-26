from datetime import datetime


def get_start_stop(yr, data_start=None):
    """
    Return (dstart, dstop) datetimes spanning year `yr`.

    Parameters
    ----------
    yr : year to process
    data_start : date, optional
        If given and its year equals `yr`, the window starts at
        data_start instead of Jan 1 — e.g. HMI science data begins
        2010-05-01, so keys-2010.fits should only cover May-Dec.
    """
    dstop = datetime(yr + 1, 1, 1)
    if data_start is not None and data_start.year == yr:
        dstart = datetime(data_start.year, data_start.month, data_start.day)
    else:
        dstart = datetime(yr, 1, 1)
    return dstart, dstop
