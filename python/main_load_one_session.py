import sys,os
import neuroseries as nts
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
# from wrappers import *
# from functions import *
import sys
import scipy.signal
from pycircstat.descriptive import mean as circmean
from matplotlib.colors import hsv_to_rgb
from wrappers import *
from functions import *
from scipy.ndimage.filters import gaussian_filter

path = '/mnt/DataAdrienBig/PeyracheLabData/Sofia/A6500/A6504/A6504-210408'
name = path.split('/')[-1]
			
A, C, position 	= loadCalciumData(path)

DFF 			= C.diff()
DFF 			= DFF.fillna(0).as_dataframe()
DFF[DFF<0]		= 0

tuningcurve		= computeCalciumTuningCurves(DFF, position['ry'], norm=True)
tuningcurve 	= smoothAngularTuningCurves(tuningcurve)			


pf,extent 				= computePlaceFields(DFF, position[['x', 'z']], nb_bins = 20)

figure()
for i, n in enumerate(tuningcurve.columns):
	subplot(10,14,i+1,projection='polar')
	plot(tuningcurve[n])
	xticks([])
	yticks([])



dims = (304,304)


af = np.zeros((A.shape[0], dims[0], dims[1]))

for i in range(A.shape[0]):
	af[i] = A[i]


peaks = pd.Series(index=tuningcurve.columns,data = np.array([circmean(tuningcurve.index.values, tuningcurve[i].values) for i in tuningcurve.columns]))

H = peaks.values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)

colorA = np.zeros((dims[0], dims[1], 3))
colorA *= np.nan

for i in range(len(af)):
	colorA[af[i].T > 3] = RGB[i]

figure()
imshow(colorA)

figure()
for i in range(len(pf)):
	subplot(10,14,i+1)	
	tmp = gaussian_filter(pf[i], 2)
	imshow(tmp, extent = extent, cmap = 'jet')
	xticks([])
	yticks([])