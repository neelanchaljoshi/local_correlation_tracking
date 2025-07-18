seriesname = 'hmi.ic_45s'  # HMI Continuum Intensity, 45s cadence
outdir = '/scratch/seismo/joshin/pipeline-test/IterativeLCT/hmi.ic_45s'
cadence = 45  # in seconds

QbitsPass = 0b00000000000000000000000000000000

KeyList = [ 
    ('t_rec', bytes),
    ('t_obs', bytes),
    ('quality', str),
    ('crpix1', float),
    ('crpix2', float),
    ('crval1', float),
    ('crval2', float),
    ('cdelt1', float),
    ('cdelt2', float),
    ('crota2', float),
    ('crln_obs', float),
    ('crlt_obs', float),
    ('rsun_obs', float),
]