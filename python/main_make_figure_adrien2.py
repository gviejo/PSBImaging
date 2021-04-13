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

# data_directory = '/mnt/DataAdrienBig/PeyracheLabData/Guillaume'
data_directory = '/mnt/DataRAID/MINISCOPE'
#datasets = np.loadtxt('/home/guillaume/PSBImaging/python/datasets_SAMERIGIDBODY.txt', delimiter = '\n', dtype = str, comments = '#')
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_SAMERIGIDBODY.txt', comment = '#', header = None)
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
	print(i, s)
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




# ########################################################################################
# # LOAD CELL REG SESSIONS PAIRS
# ########################################################################################
# path = '/mnt/DataRAID/MINISCOPE/A0600/A0634/CellRegPairs'
# files = os.listdir(path)	
# cellreg_file = np.sort([f for f in files if 'cellRegistered' in f])

# cellregdict = dict()
# count = 0
# for i in range(len(cellreg_file)):
# 	sess_pair = cellreg_file[i].split('.')[0].split('_')[-2:]
# 	ses = np.array(sess_pair).astype(np.int)	
# 	if np.diff(ses)[0] == 1:
# 		count += 1
# 		arrays = {}
# 		f = h5py.File(os.path.join(path, cellreg_file[i]))
# 		for k, v in f.items():
# 		    arrays[k] = v
# 		cellreg = np.copy(np.array(arrays['cell_registered_struct']['cell_to_index_map']))
# 		f.close()
# 		cellreg = cellreg.T - 1
# 		cellreg = cellreg.astype(np.int)
# 		idxreg = np.where(np.sum(cellreg!=-1, 1) == 2)[0]

# 		df = pd.DataFrame(index = cellreg[idxreg][:,0], columns = cellreg[idxreg][:,1], data = np.eye(len(idxreg)).astype(np.int))
# 		cellregdict[count-1] = df


		
# newcellreg = {}
# for i in cellregdict.keys(): # sessions	
# 	tmp = cellregdict[i]
# 	tmp = tmp.loc[(tmp.sum(1)==1).values,(tmp.sum(1)==1).values]
# 	idx = tmp.index.values
# 	newcellreg[i] = []
# 	for j in idx: # neurons
# 		nxt = tmp.loc[j].idxmax()
# 		chain = [j, nxt]
# 		for k in range(i+1, count): # next sessions
# 			tmp2 = cellregdict[k]
# 			tmp2 = tmp2.loc[(tmp2.sum(1)==1).values,(tmp2.sum(1)==1).values]			
# 			idx2 = tmp2.index.values
# 			if nxt in idx2:	
# 				nxt2 = tmp2.loc[nxt].idxmax()
# 				chain.append(nxt2)
# 				cellregdict[k].loc[nxt,nxt2] = 0
# 				nxt = nxt2			
# 			else:
# 				break
# 		newcellreg[i].append(chain)


# cellreg2 = []
# for i in newcellreg.keys():
# 	tmp = np.ones((len(newcellreg[i]),count+1))*-1
# 	for j,l in enumerate(newcellreg[i]):
# 		tmp[j,i:i+len(l)] = l
# 	cellreg2.append(tmp)

# cellreg = np.vstack(cellreg2)

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




####################################################################################
# DIFF PEAKS + SESSIONS
####################################################################################
diffpeaks = computePeaksAngularDifference(alltc, sessions = np.arange(len(SF)))
diffsess = computePairwiseAngularDifference(alltc, sessions = np.arange(len(SF)))



####################################################################################
# RANDOMIZING 
####################################################################################
rnddiffsess = []
for k in range(100):
	print(k)
	rndcellreg = np.copy(cellreg[list(alltc.keys())])
	for t in range(rndcellreg.shape[1]):		
		np.random.shuffle(rndcellreg[:,t])
	rndtc = {}
	for i in range(len(rndcellreg)):
		rndtc[i] = pd.DataFrame(columns = np.arange(rndcellreg.shape[1]))
		for j in np.where(rndcellreg[i]!=-1)[0]:
			rndtc[i][j] = TC[j][rndcellreg[i,j]]

	rnddiffsess.append(computePairwiseAngularDifference(rndtc, sessions = np.arange(len(SF))))

rnddiffsess = pd.concat(rnddiffsess)

# figure()
# for i,n in enumerate(alltc.keys()):
# 	subplot(10,10,i+1, projection = 'polar')
# 	plot(alltc[n])
# 	xticks([])
# 	yticks([])
# 	title(n)


pairs = (21, 4)

from matplotlib import colors

dims = (304, 304)
xcut = (90,260)
ycut = (50,220)
index = list(alltc.keys())
toplot = [0, 1, 6]
nb_bins = 15


figure(figsize = (18, 6))

gs = GridSpec(2, len(toplot)+2, wspace = 0.4, width_ratios = [0.5, 0.5, 0.5, 0.1, 1])


count = 0
for k,i in enumerate(toplot):
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
		

	subplot(gs[0,k])
	imshow(colorA[ycut[0]:ycut[1],xcut[0]:xcut[1]])
	if i == 0: ylabel('Spatial footprints')
	title('Day '+str(i+1))
	xticks([])
	yticks([])

	subplot(gs[1,k], projection = 'polar')
	plot(alltc[pairs[0]][i], color=colors.to_rgba('orange'), linewidth = 2)
	plot(alltc[pairs[1]][i], color=colors.to_rgba('green'), linewidth = 2)
	yticks([])
	xticks([0, np.pi/2, np.pi, np.pi + np.pi/2])

subplot(gs[:,-1])

tmp = diffsess.values.astype(np.float32).flatten()
tmp2 = rnddiffsess.values.astype(np.float32).flatten()

hist(np.rad2deg(tmp), np.linspace(0, 180, nb_bins), density=True, alpha = 0.5, histtype = 'step', linewidth = 3)
hist(np.rad2deg(tmp2), np.linspace(0, 180, nb_bins), density=True, color = 'grey', alpha = 0.5, histtype = 'step', label = 'random')

xlabel('Angular difference (deg)')
ylabel('%')

yticks([0, 0.01], ['0', '1'])

title('Pairwise Angular Difference across days')
legend()


savefig('../figures/figure_adrien2.pdf', format = 'pdf', dpi = 200, bbox_inches = 'tight')