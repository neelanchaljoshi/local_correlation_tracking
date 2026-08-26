import numpy as np
from datetime import datetime
from astropy.table import Table, Column
from fetch_keys import get_info
from utils.time_helpers import get_start_stop
from settings import Config


def process_year(yr: int, cfg: Config) -> None:
    """Fetch and write one year's keys table for the given Config."""
    dstart, dstop = get_start_stop(yr, cfg.data_start)
    dspan = dstop - dstart
    nt = int(dspan.total_seconds() / cfg.cadence_seconds)

    ds = (f"{cfg.seriesname}[{dstart.strftime('%Y.%m.%d_%H:%M:%S_TAI')}/"
          f"{int(dspan.total_seconds())}s@{cfg.cadence_seconds}s]")
    start = datetime.now()
    keys, path = get_info(ds, cfg.key_list)
    print(datetime.now() - start, 'get_info', yr)

    if len(path) != nt:
        raise RuntimeError(f"acquired path length {len(path)} != expected {nt}")
    if len(keys['quality']) < nt:
        raise RuntimeError(f"quality key count ({len(keys['quality'])}) < expected ({nt})")

    quality = np.array([int(q, 16) for q in keys['quality']])
    isbad = (quality | cfg.qbits_pass) != cfg.qbits_pass

    tab = Table()
    for nam, typ in cfg.key_list:
        tab[nam] = Column(keys[nam], dtype=typ)
    tab['isbad'] = Column(isbad, dtype=bool)
    tab['path'] = Column(path, dtype=bytes)

    outfile = cfg.output_path(yr)
    tab.write(outfile, format='fits', overwrite=True)
    print(datetime.now() - start, 'output', outfile)
