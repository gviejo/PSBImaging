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
from itertools import product, combinations
from scipy.stats import norm

# data_directory = '/mnt/DataAdrienBig/PeyracheLabData/Guillaume'
data_directory = '/mnt/DataRAID/MINISCOPE'
#datasets = np.loadtxt('/home/guillaume/PSBImaging/python/datasets_SAMERIGIDBODY.txt', delimiter = '\n', dtype = str, comments = '#')
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_TALKADRIEN.txt', comment = '#', header = None)
datasets = info[0].values
envs = info[1].values

SF = {}
TC = {}
HD = {}
PK = {}
SI = {}
RL = {}
PF = {}
stab_tc = {}

thr_si = 0.3
thr_rl = 2
positions = {}
allC = {}

for i, s in enumerate(datasets):
	print(s)
	name 			= s.split('/')[-1]
	path 			= os.path.join(data_directory, s)
	if envs[i] != 'cylinder':
		A, C, position 	= loadCalciumData(path, flip_ttl = True)		
	else:
		A, C, position 	= loadCalciumData(path)

	DFF 			= C.diff()
	DFF 			= DFF.fillna(0).as_dataframe()
	DFF[DFF<0]		= 0

	tuningcurve		= computeCalciumTuningCurves(DFF, position['ry'], norm=True)
	tuningcurve 	= smoothAngularTuningCurves(tuningcurve)			
	peaks 			= pd.Series(index=tuningcurve.keys(),
		data = np.array([circmean(tuningcurve[i].index.values, tuningcurve[i].values) for i in tuningcurve.keys()]))	
	si 				= computeSpatialInfo(tuningcurve, position['ry'])		
	stat 			= computeRayleighTest(tuningcurve)	
	
	tcurves2 = []
	DFF2 = np.array_split(DFF,2)	
	for j in range(2):		
		tcurves_half	= computeCalciumTuningCurves(DFF2[j], position['ry'], norm=True)
		tcurves_half 	= smoothAngularTuningCurves(tcurves_half)	
		tcurves2.append(tcurves_half)

	# diff = np.sum(np.abs(tcurves2[0] - tcurves2[1]), 0)	
	diff = {}
	for j in tuningcurve.columns:
		diff[j] = np.corrcoef(tcurves2[0][j], tcurves2[1][j])[0,1]
	diff = pd.Series(diff)	

	pf, extent			= computePlaceFields(DFF, position[['x', 'z']], 15)
	

	# tokeep 			= si[(si>thr_si).values].index.values
	tokeep 			= stat[(stat['z']>thr_rl).values].index.values	
	SI[i] 			= si
	SF[i] 			= A
	TC[i] 			= tuningcurve
	HD[i] 			= tokeep
	PK[i] 			= peaks
	RL[i]			= stat
	stab_tc[i]	 	= diff
	PF[i] 			= pf
	positions[i] 	= position
	allC[i] 		= C

cellreg, scores = loadCellReg(os.path.join(data_directory, '/'.join(datasets[0].split('/')[0:-1])))

n_sessions_detected = np.sum(cellreg!=-1, 1)

scores = pd.Series(index = np.where(n_sessions_detected>1)[0], data = scores[np.where(n_sessions_detected>1)[0]])

allst = {}
for i in np.where(n_sessions_detected>1)[0]:
	allst[i] = pd.Series(index = np.arange(cellreg.shape[1]))
	for j in np.where(cellreg[i]!=-1)[0]:		
		allst[i][j] = stab_tc[j].loc[cellreg[i,j]]		

allst = pd.concat(allst, 1).T

# Selecting neurons with stable tuning curves
allst[allst<0.4] = np.nan
# Selectingneurons with good cellreg scores
allst = allst.loc[scores[scores>0.8].index.values]
tokeep = allst.index.values
scores = scores.loc[tokeep]

alltc = {}
allsi = {}
allpf = {}
for i in tokeep:
	alltc[i] = pd.DataFrame(columns = np.arange(cellreg.shape[1]))
	allsi[i] = pd.Series(index = np.arange(cellreg.shape[1]))	
	allpf[i] = np.zeros((cellreg.shape[1], PF[0].shape[1], PF[0].shape[2],))
	for j in np.where(cellreg[i]!=-1)[0]:
		alltc[i][j] = TC[j][cellreg[i,j]]
		allsi[i][j] = SI[j].loc[cellreg[i,j]]		
		allpf[i][j] = PF[j][cellreg[i,j]]

from matplotlib import colors

dims = (304, 304)
xcut = (90,260)
ycut = (50,200)
index = list(alltc.keys())
pairs = (index[2], index[1])
titles = ['Cylinder', 'Square']
figure(figsize = (17, 8))

gs = GridSpec(2, 4, wspace = 0.4)
for i in range(2):
	subplot(gs[i, 0])
	plot(positions[i]['x'], positions[i]['z'])
	xlabel('x')
	ylabel('y')
	xticks([])
	yticks([])
	if i == 0: title('Environment')

for i in range(2):
	colorA = np.ones((dims[0], dims[1],4))
	for j, n in enumerate(range(len(SF[i]))):
		if n == cellreg[pairs,i][0]:
			colorA[SF[i][n].T > 2] = colors.to_rgba('orange')
		if n == cellreg[pairs,i][1]:
			colorA[SF[i][n].T > 2] = colors.to_rgba('green')	
		if n in cellreg[index,i] and n not in cellreg[pairs,i]:
			colorA[SF[i][n].T > 4] = colors.to_rgba('lightsteelblue')
		elif n not in cellreg[index,i]:
			colorA[SF[i][n].T > 6] = colors.to_rgba('lightgrey')
		

	subplot(gs[i,1])
	imshow(colorA[ycut[0]:ycut[1],xcut[0]:xcut[1]])
	if i == 0: title('Spatial footprints')
	xticks([])
	yticks([])

for i in range(2):
	subplot(gs[i,2])
	for j, n in enumerate(cellreg[index[0:20],i]):
		start = C.index.values[0]
		tmp = allC[i][n].loc[start:start+120*1e6].as_units('s')
		if n == cellreg[pairs,i][0]:
			plot(tmp.index.values, tmp.values+j*5, linewidth = 3, color = colors.to_rgba('orange'))
		elif n == cellreg[pairs,i][1]:
			plot(tmp.index.values, tmp.values+j*5, linewidth = 3, color = colors.to_rgba('green'))
		else:
			plot(tmp.index.values, tmp.values+j*5, color = colors.to_rgba('lightsteelblue'))
	if i == 0: title("Calcium traces")
	if i == 1: xlabel('Time (s)')
	ylabel("Neurons")
	xticks([100,160,220], [0, 60, 120])

for i in range(2):
	subplot(gs[i,3], projection = 'polar')
	plot(alltc[pairs[0]][i], color=colors.to_rgba('orange'), linewidth = 2)
	plot(alltc[pairs[1]][i], color=colors.to_rgba('green'), linewidth = 2)
	yticks([])
	xticks([0, np.pi/2, np.pi, np.pi + np.pi/2])

savefig('../figures/figure_adrien.pdf', format = 'pdf', dpi = 200, bbox_inches = 'tight')