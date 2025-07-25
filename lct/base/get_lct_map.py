#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:56:34 2023

@author: joshin
"""

import numpy as np

def get_lct_map(patch1, patch2, patch_size, gkern):

	patch1 = gkern*(patch1 - np.nanmean(patch1))
	patch2 = gkern*(patch2 - np.nanmean(patch2))
	f = np.fft.rfft2(patch1)
	g = np.fft.rfft2(patch2)
	c = np.conj(f)*g

	ccf = np.fft.irfft2(c, s= (patch_size, patch_size), axes = [0,1])
	ccf = np.fft.fftshift(ccf, axes = [0,1])
	return ccf, patch1, patch2