#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:14:33 2023

@author: joshin
"""

import numpy as np
from scipy import signal

def tukey_twoD(width, alpha):
	"""2D tukey lowpass window with a circular support
	"""
	base = np.zeros((width, width))
	tukey = signal.tukey(width, alpha)
	tukey = tukey[int(len(tukey)/2)-1:]  # Second half of tukey window
	x = np.linspace(-width/2, width/2, width)
	y = np.linspace(-width/2, width/2, width)
	for x_index in range(0, width):
		for y_index in range(0, width):
			# Only plot tukey value with in circle of radius width
			if int(np.sqrt(x[x_index]**2 + y[y_index]**2)) <= width/2:
				base[x_index, y_index] = tukey[int(np.sqrt(x[x_index]**2
					 + y[y_index]**2))]
					# Based on k**2 find tukey window value and place in matrix
	return base