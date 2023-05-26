#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:04:13 2023

@author: joshin
"""

import numpy as np


def get_ux_uy_ellipsoid(clat, clng, ccf_av, patch_size, R_sun, pixel_size, cadence):
	ym, xm = np.where(ccf_av==ccf_av[1:-1, 1:-1].max())
	xmax = xm[0]
	ymax = ym[0]

	f00 = ccf_av[ymax, xmax]
	f10 = ccf_av[ymax, xmax+1]
	f11 = ccf_av[ymax+1, xmax+1]
	f01 = ccf_av[ymax+1, xmax]
	f_11 = ccf_av[ymax+1, xmax-1]
	f_10 = ccf_av[ymax, xmax-1]
	f_1_1 = ccf_av[ymax-1, xmax-1]
	f0_1 = ccf_av[ymax-1, xmax]
	f1_1 = ccf_av[ymax-1, xmax+1]



	fxy = [[f00], [f10], [f11], [f01], [f_11], [f_10], [f_1_1], [f0_1], [f1_1]]

	A = np.array([[0,0,0,0,0,1], [1,0,1,0,0,1], [1,1,1,1,1,1], [0,1,0,1,0,1], [1,1,-1,1,-1,1], [1,0,-1,0,0,1], [1,1,-1,-1,1,1], [0,1,0,-1,0,1], [1,1,1,-1,-1,1]])
	B = np.matmul(A.T, A)
	Binv = np.linalg.inv(B)
	complete = np.matmul(Binv, A.T)


	A = np.matmul(complete, fxy)

	a,b,c,d,e,f = A

	a,b,c,d,e = a[0], b[0], c[0], d[0], e[0]

	ypar = ((e*c)-(2*a*d))/((4*a*b-(e**2)))
	xpar= (-e*ypar - c)/(2*a)

	del_x = xmax  - patch_size//2 + xpar
	del_y = ymax  - patch_size//2 + ypar

	ux = (R_sun * pixel_size * (np.pi/180) * del_x)/(cadence) * 1e6
	uy = (R_sun * del_y * pixel_size * (np.pi/180))/(cadence) * 1e6


	return del_x, del_y, ux, uy