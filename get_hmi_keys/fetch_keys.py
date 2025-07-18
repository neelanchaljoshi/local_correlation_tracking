import subprocess as subp
import numpy as np

def get_info(ds, keylist):
    knames = ','.join([nam for nam, _ in keylist])
    p = subp.Popen(f'show_info ds={ds} key={knames} -q', shell=True, stdout=subp.PIPE, encoding='utf-8')
    lines = [line.strip() for line in p.stdout.readlines()]
    keys_str = np.array([line.split() for line in lines])

    keys = {}
    for i, (nam, typ) in enumerate(keylist):
        keys[nam] = keys_str[:, i].astype(typ)

    path = subp.Popen(f'show_info ds={ds} -Pq', shell=True, stdout=subp.PIPE, encoding='utf-8').stdout.readlines()
    return keys, path