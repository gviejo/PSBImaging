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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import product, combinations
from scipy.stats import norm
from scipy.ndimage.filters import gaussian_filter

#dims = (304,304)
#dims = (202, 216)
dims = (178, 158) # A0634

data_directory = '/mnt/DataRAID/MINISCOPE'

fbasename = 'A0634'
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
paths = [os.path.join(data_directory, fbasename[0:3] + '00', fbasename, 'minian', fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
sessions = [fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '') for i in info.index]
info['paths'] = paths
info['sessions'] = sessions
info = info.set_index('sessions')

#############################################################################################
# Cell Registration
#############################################################################################
cellreg = pd.read_csv(os.path.join(data_directory, fbasename[0:3] + '00', fbasename, 'minian', 'minian_reg.csv'), index_col = [0])

# intersecting info and cellreg sessions
sessions = np.intersect1d(info.index.values, cellreg.columns.values)

info = info.loc[sessions]
cellreg = cellreg[sessions]

n_sessions_detected = cellreg.notna().sum(1)

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



for i, s in enumerate(sessions):
	path 			= info.loc[s,'paths']	
	if os.path.exists(path):
		print(path)
		name 			= os.path.basename(path)		
		A, C, position 	= loadCalciumData(path, dims = dims)
		print(A.shape)

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




figure()
count = 1
for i in SF.keys():
	ax = subplot(int(np.sqrt(len(SF)))+1,int(np.sqrt(len(SF)))+1,count)
	# a = (SF[i]>np.percentile(SF[i],0.001)).sum(0)	
	imshow(SF[i].sum(0), cmap = 'viridis')
	count += 1


cellreg = cellreg.fillna(-1).astype(np.int).values

tokeep = np.where(n_sessions_detected > 1)[0]

allst = {}
for i in tokeep:
	allst[i] = pd.Series(index = np.arange(cellreg.shape[1]))
	for j in np.where(cellreg[i]!=-1)[0]:		
		allst[i][j] = stab_tc[j].loc[cellreg[i,j]]		

allst = pd.concat(allst, 1).T



# Selecting neurons with stable tuning curves
allst[allst<0.3] = np.nan

tokeep = allst[allst.notna().any(1)].index.values




alltc = {}
allsi = {}
allpf = {}
for i in tokeep:
	alltc[i] = pd.DataFrame(columns = sessions)
	allsi[i] = pd.Series(index = sessions)	
	allpf[i] = np.zeros((len(sessions), PF[0].shape[1], PF[0].shape[2],))
	for j in np.where(cellreg[i]!=-1)[0]:
		alltc[i][sessions[j]] = TC[j][cellreg[i,j]]
		allsi[i].loc[sessions[j]] = SI[j].loc[cellreg[i,j]].values
		allpf[i][j] = PF[j][cellreg[i,j]]

#####################################################################################
# PLOT ALL TUNING CURVES
#####################################################################################
index = np.array(list(alltc.keys()))[0:20]

rigs = ['Circular', 'Square', '8-arm maze', 'Open field']

figure()
for i, n in enumerate(index):	
	ax = subplot(int(np.ceil(np.sqrt(len(index))))-1,int(np.ceil(np.sqrt(len(index)))),i+1)
	gs = GridSpecFromSubplotSpec(1,len(rigs),ax)
	tmp = alltc[n].dropna(1, 'all')
	grp = info.loc[tmp.columns].groupby('Rig').groups
	for j, m in enumerate(rigs):
		if m in grp.keys():
			subplot(gs[0,j], projection = 'polar')
			plot(alltc[n][grp[m]], color = cm.rainbow(np.linspace(0,1,len(rigs)))[j])
			xticks([])
			yticks([])

sys.exit()

# figure()
# index = alltc.keys()
# for i, n in enumerate(index):
# 	subplot(int(np.ceil(np.sqrt(len(index)))),int(np.ceil(np.sqrt(len(index)))),i+1, projection='polar')
# 	for j in alltc[n].columns:
# 		plot(alltc[n][j])
# 	xticks([])
# 	yticks([])


# figure()
# index = list(alltc.keys())
# gs = GridSpec(int(np.ceil(np.sqrt(len(index)))),int(np.ceil(np.sqrt(len(index)))))
# count = 0
# for i in range(int(np.ceil(np.sqrt(len(index))))):
# 	for j in range(int(np.ceil(np.sqrt(len(index))))):
# 		gs2 = gs[i,j].subgridspec(1,2)
# 		n = index[count]
# 		subplot(gs2[0,0], projection='polar')
# 		for k in alltc[n].columns:
# 			plot(alltc[n][k])
# 		xticks([])
# 		yticks([])
# 		tmp = []		
# 		for k in range(len(allpf[n])):
# 			tmp.append(gaussian_filter(allpf[n][k], 2))
# 		tmp = np.array(tmp)
# 		gs3 = gs2[0,1].subgridspec(2,2)
# 		xp, yp = ([0,0,1,1],[0,1,0,1])
# 		for k in range(len(allpf[n])):
# 			subplot(gs3[xp[k],yp[k]])
# 			imshow(tmp[k], cmap = 'jet')#, vmin = tmp.min(), vmax = tmp.max())
# 			xticks([])
# 			yticks([])
# 		count += 1
# 		if count == len(index):
# 			break
# 	if count == len(index):
# 		break







####################################################################################
# DIFF PEAKS + SESSIONS
####################################################################################
diffpeaks = computePeaksAngularDifference(alltc, sessions = sessions)
diffsess = computePairwiseAngularDifference(alltc, sessions = sessions )



####################################################################################
# RANDOMIZING 
####################################################################################
rnddiffsess = []
for k in range(20):
	print(k)
	rndcellreg = np.copy(cellreg[list(alltc.keys())])
	for t in range(rndcellreg.shape[1]):
		np.random.shuffle(rndcellreg[:,t])
	rndtc = {}
	for i in range(len(rndcellreg)):
		rndtc[i] = pd.DataFrame(columns = sessions)
		for j in np.where(rndcellreg[i]!=-1)[0]:
			rndtc[i][sessions[j]] = TC[j][rndcellreg[i,j]]

	rnddiffsess.append(computePairwiseAngularDifference(rndtc, sessions = sessions))

rnddiffsess = pd.concat(rnddiffsess)




#####################################################################################
# CELL REG ANGULAR DIFFERENCE OF PREFERRED DIRECTION STARTING AT FIRST SESSION
#####################################################################################
# index = tokeep[np.where(n_sessions_detected[tokeep] == len(SF))[0]]
envs = info.loc[sessions].groupby('Rig').groups

sinter = []
for a, b  in combinations(envs.keys(), 2):
	for p in product(envs[a], envs[b]):
		sinter.append(tuple(np.sort(p)))

swithin = {s:list(combinations(envs[s], 2)) for s in envs.keys()}


figure()
nb_bins = 10
clrs = ['blue', 'red', 'green']
for i, e in enumerate(swithin.keys()):
	subplot(2,2,i+1)
	tmp = diffsess[swithin[e]].values.astype(np.float32).flatten()
	tmp2 = rnddiffsess[swithin[e]].values.astype(np.float32).flatten()
	title(e)
	hist(np.rad2deg(tmp), np.linspace(0, 180, nb_bins), density=True, color = clrs[0], alpha = 0.5, histtype = 'step')
	hist(np.rad2deg(tmp2), np.linspace(0, 180, nb_bins), density=True, color = 'grey', alpha = 0.5, histtype = 'step')
	xlim(0, 180)
subplot(224)
tmp = diffsess[sinter].values.astype(np.float32).flatten()
tmp2 = rnddiffsess[sinter].values.astype(np.float32).flatten()
title('Inter')
hist(np.rad2deg(tmp), np.linspace(0, 180, nb_bins), density=True, color = clrs[2], alpha = 0.5, histtype = 'step')
hist(np.rad2deg(tmp2), np.linspace(0, 180, nb_bins), density=True, color = 'grey', alpha = 0.5, histtype = 'step')
xlim(0, 180)


