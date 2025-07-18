import os
import numpy as np
from datetime import datetime
from astropy.table import Table, Column
from fetch_keys import get_info
from utils.time_helpers import get_start_stop
from config import cadence, seriesname, outdir, QbitsPass, KeyList

def process_year(yr):
    dstart, dstop = get_start_stop(yr)
    dspan = dstop - dstart
    nt = int(dspan.total_seconds() / cadence)

    ds = f"{seriesname}[{dstart.strftime('%Y.%m.%d_%H:%M:%S_TAI')}/{int(dspan.total_seconds())}s@{cadence}s]"
    start = datetime.now()
    keys, path = get_info(ds, KeyList)
    print(datetime.now() - start, 'get_info', yr)

    if len(path) != nt:
        raise RuntimeError(f"acquired path length {len(path)} != expected {nt}")
    if len(keys['quality']) < nt:
        raise RuntimeError(f"quality key count ({len(keys['quality'])}) < expected ({nt})")

    quality = np.array([int(q, 16) for q in keys['quality']])
    isbad = (quality | QbitsPass) != QbitsPass

    tab = Table()
    for nam, typ in KeyList:
        tab[nam] = Column(keys[nam], dtype=typ)
    tab['isbad'] = Column(isbad, dtype=bool)
    tab['path'] = Column(path, dtype=bytes)

    outfile = os.path.join(outdir, f'keys-{yr}.fits')
    tab.write(outfile, format='fits', overwrite=True)
    print(datetime.now() - start, 'output', outfile)