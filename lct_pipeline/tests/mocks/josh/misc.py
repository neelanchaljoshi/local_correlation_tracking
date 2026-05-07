from datetime import datetime, timedelta
import string

def xdays(start, stop, step):
    t = start
    while t < stop:
        yield t
        t += step

def strip_nonprintable(s):
    printable = set(string.printable)
    return ''.join(c for c in s if c in printable)
