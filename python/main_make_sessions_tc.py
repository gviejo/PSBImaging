import sys,os
import neuroseries as nts
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys
import scipy.signal
from pycircstat.descriptive import mean as circmean
from matplotlib.colors import hsv_to_rgb
import cv2
from matplotlib.gridspec import GridSpec

data_directory = '/mnt/DataAdrienBig/PeyracheLabData/Guillaume'
datasets = np.loadtxt('/home/guillaume/PSBImaging/python/datasets_PSB.txt', delimiter = '\n', dtype = str, comments = '#')

SF = {}
TC = {}
HD = {}
PK = {}
SI = {}

thr_si = 0.2

for i, s in enumerate(datasets):
	print(s)
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	A, C, position 	= loadCalciumData(path)		
	DFF 			= C.diff()
	DFF 			= DFF.fillna(0).as_dataframe()
	DFF[DFF<0]		= 0
	tuningcurve		= computeCalciumTuningCurves(DFF, position['ry'], norm=True)
	tuningcurve 	= smoothAngularTuningCurves(tuningcurve)	
	peaks 			= pd.Series(index=tuningcurve.keys(),
		data = np.array([circmean(tuningcurve[i].index.values, tuningcurve[i].values) for i in tuningcurve.keys()]))
	# tokeep, stat 	= findHDCells(tuningcurve, z=1, p = 0.05)
	si 				= computeSpatialInfo(tuningcurve, position['ry'])		
	tokeep 			= si[(si>thr_si).values].index.values
	SI[i] 			= si
	SF[i] 			= A
	TC[i] 			= tuningcurve
	HD[i] 			= tokeep
	PK[i] 			= peaks


cellreg = loadCellReg(os.path.join(data_directory, '/'.join(datasets[0].split('/')[0:-1])))

# Angular differences
n_sessions_detected = np.sum(cellreg!=-1, 1)

alltc = {}
allsi = {}
for i in np.where(n_sessions_detected>1)[0]:
	alltc[i] = pd.DataFrame(columns = np.arange(cellreg.shape[1]))
	allsi[i] = pd.Series(index = np.arange(cellreg.shape[1]))
	for j in np.where(cellreg[i]!=-1)[0]:
		alltc[i][j] = TC[j][cellreg[i,j]]
		allsi[i][j] = SI[j].loc[cellreg[i,j]]



idx = np.argsort(n_sessions_detected)[::-1]
figure()
for i in range(12):
	subplot(3,4,i+1, projection = 'polar')
	plot(alltc[idx[i]])




dims = (304, 304)
xcut = (90,230)
ycut = (50,200)
#####################################################################################
# PLOT SPATIAL FOOTPRINTS WITH PREFERED DIRECTION
#####################################################################################
AllA = []

figure()
for i in range(len(SF)):
	subplot(2,3,i+1)
	peaks = PK[i][HD[i]]
	H2 = peaks.values/(2*np.pi)
	HSV2 = np.vstack((H2, np.ones_like(H2), np.ones_like(H2))).T
	RGB2 = hsv_to_rgb(HSV2)
	colorA = np.ones((dims[0], dims[1], 3))
	for j, n in enumerate(peaks.index.values):
		colorA[SF[i][n].T > 3] = RGB2[j]

	nonhd = list(set(np.arange(len(SF[i]))) - set(HD[i]))
	for j, n in enumerate(nonhd):
		colorA[SF[i][n].T > 4] = [0,0,0]

	imshow(colorA[ycut[0]:ycut[1],xcut[0]:xcut[1]])

	for n in peaks.index:
		_ , _, _ , maxloc = cv2.minMaxLoc(SF[i][n].T[ycut[0]:ycut[1],xcut[0]:xcut[1]])
		annotate(str(n), (maxloc[0], maxloc[1]), (maxloc[0]+1+np.random.rand(), maxloc[1]-1-np.random.rand()), fontsize = 10)
	
	AllA.append(colorA[ycut[0]:ycut[1],xcut[0]:xcut[1]])


#####################################################################################
# PLOT CONCATENATED SPATIAL FOOTPRINT WITH CELL REG OUTPUT
#####################################################################################

AllA = np.concatenate(AllA, 1)
idx = np.argsort(n_sessions_detected)[::-1]

figure()
imshow(AllA)
for i in idx[0:3]:
	x = []
	y = []
	for j in range(len(cellreg[i])):
		if cellreg[i,j] > -1:
			tmp = SF[j][cellreg[i,j]].T[ycut[0]:ycut[1],xcut[0]:xcut[1]]
			_ , _, _ , maxloc = cv2.minMaxLoc(tmp)
			x.append(maxloc[0]+tmp.shape[1]*j)
			y.append(maxloc[1])
		plot(x, y, '.-', alpha = 0.5, linewidth = 1)
title("CellREG")


#####################################################################################
# CELL REG ANGULAR DIFFERENCE OF PREFERRED DIRECTION STARTING AT FIRST SESSION
#####################################################################################

idx = np.argsort(n_sessions_detected)[::-1]

sys.exit()

diffpeaks = pd.DataFrame(index = alltc.keys(), columns = np.arange(len(SF)))
for n in alltc.keys():
	tmp = np.where(allsi[n]>0.2)[0]
	if len(tmp>1):
		refpeak = circmean(alltc[n].index.values, alltc[n][tmp[0]].values)
		diffpeaks.loc[n,tmp[0]] = 0.0
		for k in tmp[1:]:
			peak = circmean(alltc[n].index.values, alltc[n][k].values)
			diffpeaks.loc[n,k] = np.abs(np.arctan2(np.sin(peak - refpeak), np.cos(peak - refpeak)))

diffpeaks = diffpeaks.astype(np.float32)

figure()
gs = GridSpec(2,len(SF)+1)
n = idx[0]
for i in range(len(SF)):
	subplot(gs[0,i+1],projection = 'polar')
	plot(alltc[n][i])
	if i == 3:
		title("CellREG longest alignement")
	subplot(gs[1,1:])	
	tmp = diffpeaks[~diffpeaks.isna().all(1)]
	for j in tmp.index.values:
		plot(tmp.loc[j].dropna())
	title('Diff consecutive peak')
subplot(gs[1,0])
tmp = diffpeaks.values
tmp2 = tmp[~np.isnan(tmp)]
hist(tmp2[tmp2>0], orientation = 'horizontal')


#####################################################################################
# CELL REG ANGULAR DIFFERENCE OF PAIRS OF NEURONS
#####################################################################################
idx = np.argsort(n_sessions_detected)[::-1]

from itertools import combinations
# pairs detected in the same sessions
pairs = list(combinations(idx, 2))
pairs_together = {}
for p in pairs:
	sessions_together = np.prod((cellreg[p,:]>-1)*1, 0)
	if np.sum(sessions_together) > 1:
		pairs_together[p] = np.where(sessions_together)[0]

pairs_diff = {}
for p in pairs_together.keys():
	si = pd.concat((allsi[p[0]][pairs_together[p]],allsi[p[1]][pairs_together[p]]),1)
	ses = si[(si>thr_si).all(1)]
	tmp = ses.index.values
	if len(tmp)>1:
		tc0 = alltc[p[0]][tmp]
		tc1 = alltc[p[1]][tmp]
		pk = pd.concat((
			pd.Series(index=tc0.keys(), data = np.array([circmean(tc0[i].index.values, tc0[i].values) for i in tc0.keys()])),
			pd.Series(index=tc1.keys(), data = np.array([circmean(tc1[i].index.values, tc1[i].values) for i in tc1.keys()]))	
			), 1)
		angdiff = np.arctan2(np.sin(pk[1] - pk[0]), np.cos(pk[1] - pk[0]))
		pairs_diff[p] = angdiff


gs = GridSpec(3,2, width_ratios = [0.2, 0.8])
figure()
n = idx[0]
for i in range(len(SF)):
	subplot(gs[0,1])	
	tmp = diffpeaks[~diffpeaks.isna().all(1)]
	for j in tmp.index.values:
		plot(tmp.loc[j].dropna())
	title('Diff consecutive peak')
subplot(gs[0,0])
tmp = diffpeaks.values
tmp2 = tmp[~np.isnan(tmp)]
hist(tmp2[tmp2>0], orientation = 'horizontal')

meanalpha = []
subplot(gs[1,1])
for p in pairs_diff:
	plot(np.abs(pairs_diff[p]), 'o-')
title('Angular Difference / pair')
subplot(gs[2,1])
for p in pairs_diff:
	tmp = pairs_diff[p].diff()
	alpha = pd.Series(index = pairs_diff[p].index.values, data = 0)
	alpha.loc[alpha.index[1:]] = np.abs(np.arctan2(np.sin(tmp.iloc[1:]), np.cos(tmp.iloc[1:])))
	plot(alpha, 'o-')
	
	meanalpha.append(alpha.iloc[1:].mean())
title('Difference of angular difference / pair')
subplot(gs[2,0])
hist(meanalpha, orientation = 'horizontal')



#####################################################################################
# EXAMPLES
#####################################################################################
sessions = [2,3,4]
alig = np.array([
	[22, 39, 69],
	[18, 14, 9],
	[19, 17, 10],	
	])

AllAA = []
for t, i in enumerate(sessions):
	peaks = PK[i][HD[i]]
	H2 = peaks.values/(2*np.pi)
	HSV2 = np.vstack((H2, np.ones_like(H2), np.ones_like(H2))).T
	RGB2 = hsv_to_rgb(HSV2)
	colorA = np.ones((dims[0], dims[1], 4))
	for j, n in enumerate(peaks.index.values):
		if n in alig[t]:
			colorA[SF[i][n].T > 3] = np.hstack((RGB2[j], [1]))
		else:
			colorA[SF[i][n].T > 3] = np.hstack((RGB2[j], [0.35]))
	nonhd = list(set(np.arange(len(SF[i]))) - set(HD[i]))
	for j, n in enumerate(nonhd):
		colorA[SF[i][n].T > 4] = [0,0,0,0.35]	
	AllAA.append(colorA[ycut[0]:ycut[1],xcut[0]:xcut[1]])

AllAA = np.concatenate(AllAA, 1)


figure()
gs = GridSpec(2,3)
subplot(gs[0,:])
imshow(AllAA, aspect = 'auto')
for i in range(len(alig)):
	x = []
	y = []
	for j in range(len(alig[i])):
		tmp = SF[sessions[j]][alig[i,j]].T[ycut[0]:ycut[1],xcut[0]:xcut[1]]
		_ , _, _ , maxloc = cv2.minMaxLoc(tmp)
		x.append(maxloc[0]+tmp.shape[1]*j)
		y.append(maxloc[1])
	plot(x, y, '.-', alpha = 0.5, linewidth = 1)

for i,s in enumerate(sessions):
	subplot(gs[1,i], projection = 'polar')
	for j in range(len(alig)):
		peaks = PK[s][alig[j,i]]
		H2 = peaks/(2*np.pi)
		HSV2 = np.vstack((H2, np.ones_like(H2), np.ones_like(H2))).T
		RGB2 = hsv_to_rgb(HSV2)

		plot(TC[s][alig[j,i]], color = RGB2[0])

