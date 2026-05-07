# Local Correlation Tracking — SDO/HMI Solar Flow Analysis

A parallelised Python pipeline for tracking surface flows on the Sun using
Local Correlation Tracking (LCT) applied to SDO/HMI image data. Developed
as part of doctoral research at the Max Planck Institute for Solar System
Research, Göttingen.

![Tests](https://github.com/neelanchaljoshi/local_correlation_tracking/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/neelanchaljoshi/local_correlation_tracking/branch/main/graph/badge.svg)](https://codecov.io/gh/neelanchaljoshi/local_correlation_tracking)

## Overview

This repository contains two production-level Python pipelines developed
as part of doctoral research on solar surface flows and inertial modes.
The pipelines process large volumes of solar image data (~1TB/year) to
extract weak surface flow signals at the limits of instrument sensitivity.

### `lct_pipeline/`
End-to-end LCT pipeline for tracking granulation and magnetic features on
the solar surface using SDO/HMI continuum and magnetogram images. Supports
MPI parallelisation for HPC cluster execution via SLURM, configurable for
both granulation and magnetic feature tracking via `.ini` config files.

### `inertial_mode_pipeline/`
Pipeline for extracting solar inertial mode eigenfunctions from LCT flow
maps. Takes processed flow data as input, applies Fourier analysis and
SVD-based eigenfunction extraction, projects onto Legendre polynomials,
filters noise modes, and estimates errors via Monte Carlo methods.

## Key Applications

- Tracking granulation and magnetic features on the solar surface
- Extracting horizontal velocity eigenfunctions of solar inertial modes
- Synthetic parameter analysis for the ESA Vigil/PMI instrument

## Dependencies

- Python 3.9+
- NumPy, SciPy, Matplotlib, Astropy, h5py
- MPI4py (for parallelisation)
- SLURM (for HPC cluster execution)

## Usage

**LCT pipeline** (granulation tracking, one month per SLURM array job):
```bash
sbatch --array=1-12 lct_pipeline/run_slurm.sh lct_pipeline/config/granulation.ini 2019
```

**Inertial mode pipeline** (eigenfunction extraction for one mode):
```bash
python inertial_mode_pipeline/run_pipeline.py 2 -171.0 highlat hmi.m_720s_dt_1h sym \
    --l_max 22 --l_cutoff 15 --mc_samples 500
```

**Run tests:**
```bash
cd lct_pipeline && python -m pytest tests/ -v
cd inertial_mode_pipeline && python -m pytest tests/ -v
```

## Related Publications

- Joshi, N., Liang, Z.-C., Fournier, D., et al., "Horizontal velocity
  eigenfunctions of solar inertial modes using local correlation tracking
  of magnetic features", *Astronomy & Astrophysics* (under review), 2026.
- Joshi, N., Liang, Z.-C., & Gizon, L., "A synthetic parameter analysis
  of correlation tracking of granulation and magnetic features for the PMI
  instrument", *Astronomy & Astrophysics* (in prep.), 2026.

## Author

**Neelanchal Joshi**

Doctoral Researcher, Max Planck Institute for Solar System Research

[neelanchaljoshi.github.io](https://neelanchaljoshi.github.io)
