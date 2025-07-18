from datetime import datetime

def get_start_stop(yr):
    if yr == 2010:
        return datetime(2010, 5, 1), datetime(2011, 1, 1)
    else:
        return datetime(yr, 1, 1), datetime(yr + 1, 1, 1)