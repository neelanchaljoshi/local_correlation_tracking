# Local Correlation Tracking — SDO/HMI Solar Flow Analysis

A parallelised Python pipeline for tracking surface flows on the Sun using
Local Correlation Tracking (LCT) applied to SDO/HMI image data. Developed
as part of doctoral research at the Max Planck Institute for Solar System
Research, Göttingen.

## Overview

This repository contains the core pipeline used to detect, extract, and
characterise solar inertial modes from continuous SDO/HMI observations.
The pipeline processes large volumes of solar image data (~1TB/year) to
extract weak surface flow signals at the limits of instrument sensitivity.

Key applications:
- Tracking granulation and magnetic features on the solar surface
- Extracting horizontal velocity eigenfunctions of solar inertial modes
- Synthetic parameter analysis for the ESA Vigil/PMI instrument

## Dependencies

- Python 3.8+
- NumPy, SciPy, Matplotlib
- Astropy
- MPI4py (for parallelisation)
- SLURM (for HPC cluster execution)

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
