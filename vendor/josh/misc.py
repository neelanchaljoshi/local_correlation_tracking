import os, shutil
from datetime import timedelta
import numpy as np

def total_seconds(dt):
    #return dt.days*24*60*60 + dt.seconds + dt.microseconds*1e-6
    assert (dt.microseconds == 0)
    return dt.days*24*60*60 + dt.seconds

def xdays(dat, end, dt=timedelta(days=1)):
    while dat < end:
        yield dat
        dat += dt

#def add_months(dat, inc):
#    assert type(inc) is int
#    new = dat.month + inc
#    if new <= 12:
#        newmonth = new
#        newyear = dat.year
#    else:
#        newmonth = new % 12
#        newyear = dat.year + new//12
#    return dat.replace(newyear, newmonth)

def add_months(dat, dmm):
    assert type(dmm) is int
    new = dat.month + dmm
    dyy, mm = divmod(new, 12)
    if mm == 0:
        mm = 12
        dyy -= 1
    yy = dat.year + dyy
    return dat.replace(yy, mm)

def xmonths(beg, end, inc=1):
    dat = beg
    while (inc > 0 and dat < end) or (inc < 0 and dat > end):
        yield dat
        dat = add_months(dat, inc)

def append_suffix_number_push_back(src):
    if not os.path.exists(src): return
    i = 0
    dest = src + '.0'
    while os.path.exists(dest):
        i += 1
        dest = src + '.' + str(i)
    shutil.move(src, dest)

def append_suffix_number_push_front(src):
    if not os.path.exists(src): return
    root, ext = os.path.splitext(src)
    if not ext[1:].isdigit():
        dest = src + '.0'
    else:
        ver = int(ext[1:])
        dest = root + '.' + str(ver+1)
    append_suffix_number_push_front(dest)
    shutil.move(src, dest)
    return

def chkdir(dir):
    # No such file or directory
    if not os.path.exists(dir):
        raise RuntimeError("%s: `%s'" % (os.strerror(os.errno.ENOENT), dir))
    # Not a directory
    if not os.path.isdir(dir):
        raise RuntimeError("%s: `%s'" % (os.strerror(os.errno.ENOTDIR), dir))
    # Permission denied
    if not os.access(dir, os.R_OK | os.W_OK | os.X_OK):
        raise RuntimeError("%s: `%s'" % (os.strerror(os.errno.EACCES), dir))

def divmodf(x, y):
    quot = np.floor(float(x)/y)
    rem = x - y*quot
    return quot, rem

# https://stackoverflow.com/questions/8689795/how-can-i-remove-non-ascii-characters-but-leave-periods-and-spaces
import string
def strip_nonprintable(s):
    printable = set(string.printable)
    return ''.join([x for x in s if x in printable])
