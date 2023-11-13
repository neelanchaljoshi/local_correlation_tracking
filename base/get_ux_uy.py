#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:04:13 2023

@author: joshin
"""

import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.linalg import lstsq

def get_ux_uy_ellipsoid(ccf_av, grid_len = 5, patch_size = 168, pixel_size = 0.03, R_sun = 696, ntry = 3, cadence = 45):
	# Finds the ux and uy from the ccf_av
	# Cadence always in seconds
	dx_tot = 0
	dy_tot = 0

	assert grid_len%2 == 1

	for n in range(ntry):
		ym, xm = np.where(ccf_av==ccf_av[grid_len:-grid_len, grid_len:-grid_len].max())
		xmax = xm[0]
		ymax = ym[0]

		fxy = ccf_av[ymax-grid_len//2:ymax+(grid_len+1)//2, xmax-grid_len//2:xmax+(grid_len+1)//2]
		val_arr = fxy.flatten()

		x = y = np.linspace(-grid_len//2+1, grid_len//2, grid_len)

		coeff_arr = []
		for j, i in [[_i, _j] for _i in x for _j in y]:
			coeff_arr.append([i**2, j**2, i, j, i*j, 1])

		coeff_array = np.array(coeff_arr)
		p, res, rnk, s = lstsq(coeff_array, val_arr)

		a,b,c,d,e,f = p

		ypar = ((e*c)-(2*a*d))/((4*a*b-(e**2)))
		xpar= (-e*ypar - c)/(2*a)

		del_x = xmax  - patch_size//2 + xpar
		del_y = ymax  - patch_size//2 + ypar

		ccf_av = shift(ccf_av,[-del_y, -del_x], mode = 'reflect')

		dx_tot = dx_tot + del_x
		dy_tot = dy_tot + del_y

	#Calculate ux, uy
	ux = R_sun * dx_tot * np.deg2rad(pixel_size)/(cadence) * 1e6
	uy = R_sun * dy_tot * np.deg2rad(pixel_size)/(cadence) * 1e6

	return dx_tot, dy_tot, ux, uy