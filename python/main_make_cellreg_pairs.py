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
datasets = np.loadtxt('/home/guillaume/PSBImaging/python/datasets_SAMERIGIDBODY.txt', delimiter = '\n', dtype = str, comments = '#')

SF = {}
TC = {}
HD = {}
PK = {}
SI = {}
RL = {}

thr_si = 0.3
thr_rl = 3

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
	si 				= computeSpatialInfo(tuningcurve, position['ry'])		
	stat 			= computeRayleighTest(tuningcurve)	
	# tokeep 			= si[(si>thr_si).values].index.values
	tokeep 			= stat[(stat['z']>thr_rl).values].index.values
	SI[i] 			= si
	SF[i] 			= A
	TC[i] 			= tuningcurve
	HD[i] 			= tokeep
	PK[i] 			= peaks
	RL[i]			= stat


########################################################################################
# LOAD CELL REG SESSIONS PAIRS
########################################################################################
path = '/mnt/DataRAID/MINISCOPE/A0600/A0634/CellRegPairs'
files = os.listdir(path)	
cellreg_file = np.sort([f for f in files if 'cellRegistered' in f])

alltc = {0:[], 1:[]}
allsi = {0:[], 1:[]}
allpk = {0:[], 1:[]}
allrl = {0:[], 1:[]}
cellregs = {}

for i in range(len(cellreg_file)):
	sess_pair = cellreg_file[i].split('.')[0][-3:]
	arrays = {}
	f = h5py.File(os.path.join(path, cellreg_file[i]))
	for k, v in f.items():
	    arrays[k] = v
	cellreg = np.copy(np.array(arrays['cell_registered_struct']['cell_to_index_map']))
	f.close()
	cellreg = cellreg.T - 1
	cellreg = cellreg.astype(np.int)	
	cellregs[sess_pair] = cellreg

	idxreg = np.where(np.sum(cellreg!=-1, 1) == 2)[0]

	names = [sess_pair+'_'+str(n) for n in idxreg]
	ses = np.array(sess_pair.split('_')).astype(np.int)
	for j in range(2):
		tc = TC[ses[j]][cellreg[idxreg,j]]
		si = SI[ses[j]].loc[cellreg[idxreg,j]]
		pk = PK[ses[j]].loc[cellreg[idxreg,j]]
		rl = RL[ses[j]].loc[cellreg[idxreg,j],'z']
		tc.columns = names
		si.index = names
		pk.index = names
		rl.index = names
		alltc[j].append(tc)
		allsi[j].append(si)
		allpk[j].append(pk)
		allrl[j].append(rl)

for j in range(2):
	alltc[j] = pd.concat(alltc[j], 1)
	allsi[j] = pd.concat(allsi[j])
	allpk[j] = pd.concat(allpk[j])
	allrl[j] = pd.concat(allrl[j])

allsi = pd.concat(allsi, 1)
allpk = pd.concat(allpk, 1)
allrl = pd.concat(allrl, 1)

tokeep = allrl[(allrl>thr_rl).all(1)].index.values


###########################################################################################
# CROSS CORR OF TUNING CURVES OF ALL NEURONS REALIGNED
###########################################################################################
sys.exit()

tokeep = alltc[0].columns

cc_tc = []

for n in tokeep:	
	tmp = []
	for i in alltc.keys():
		tmp.append(alltc[i][n].values)
	tmp = np.array(tmp).T
	tmp2 = np.tile(tmp, (3,1))
	tmp3 = np.correlate(tmp2[:,0], tmp2[:,1], 'same').reshape(3,len(tmp)).T[:,1]
	tmp3 = pd.DataFrame(index = alltc[0].index.values-np.pi, data = tmp3, columns = [n])
	cc_tc.append(tmp3)

cc_tc = pd.concat(cc_tc, 1)

diffpeaks = np.abs(cc_tc.idxmax())

pairs = list(combinations(tokeep, 2))
pairs = np.array([p for p in pairs if p[0].split('_')[0] == p[1].split('_')[0]])

cc_tc_0 = correlateTC(alltc[0], pairs)
cc_tc_1 = correlateTC(alltc[1], pairs)


diffpairs = pd.concat([cc_tc_0.idxmax().abs(), cc_tc_1.idxmax().abs()], 1)

diffsess = np.abs(np.arctan2(np.sin(diffpairs[1]-diffpairs[0]), np.cos(diffpairs[1]-diffpairs[0])))

######################################################################################################3
# RANDOMIZING HD NEURONS
######################################################################################################3
allrnddiffpeaks = {}
allrnddiffsess  = {}

for s in range(1000):
	print(s)
	rndtc = {0:[], 1:[]}
	rndsi = {0:[], 1:[]}
	rndpk = {0:[], 1:[]}

	for sess_pair in cellregs.keys():
		cellreg = cellregs[sess_pair]
		idxreg = np.where(np.sum(cellreg!=-1, 1) == 2)[0]

		ses = np.array(sess_pair.split('_')).astype(np.int)
		si0 = SI[ses[0]].loc[cellreg[idxreg,0]]
		si1 = SI[ses[1]].loc[cellreg[idxreg,1]]
		tmp = np.hstack((si0.values, si1.values))
		hdn = np.where((tmp>thr_si).all(1))[0]

		cellreg2 = np.copy(cellreg[idxreg][hdn])
		np.random.shuffle(cellreg2[:,1])

		names = [sess_pair+'_'+str(n) for n in hdn]
		for j in range(2):
			tc = TC[ses[j]][cellreg2[:,j]]
			si = SI[ses[j]].loc[cellreg2[:,j]]
			pk = PK[ses[j]].loc[cellreg2[:,j]]
			tc.columns = names
			si.index = names
			pk.index = names
			rndtc[j].append(tc)
			rndsi[j].append(si)
			rndpk[j].append(pk)

	for j in range(2):
		rndtc[j] = pd.concat(rndtc[j], 1)
		rndsi[j] = pd.concat(rndsi[j])
		rndpk[j] = pd.concat(rndpk[j])

	rndsi = pd.concat(rndsi, 1)
	rndpk = pd.concat(rndpk, 1)

	tokeep = rndsi.index.values

	rnddiffpeaks = np.abs(np.arctan2(np.sin(rndpk[1] - rndpk[0]), np.cos(rndpk[1] - rndpk[0])))

	pairs = list(combinations(tokeep, 2))
	pairs = np.array([p for p in pairs if p[0].split('_')[0] == p[1].split('_')[0]])

	d0 = rndpk.loc[pairs[:,0],0].values - rndpk.loc[pairs[:,1],0].values
	tmp0 = np.abs(np.arctan2(np.sin(d0), np.cos(d0)))
	d1 = rndpk.loc[pairs[:,0],1].values - rndpk.loc[pairs[:,1],1].values
	tmp1 = np.abs(np.arctan2(np.sin(d1), np.cos(d1)))
	rnddiffpairs = pd.DataFrame(index = pairs, data = np.vstack((tmp0, tmp1)).T)
	rnddiffsess = np.abs(np.arctan2(np.sin(rnddiffpairs[1]-rnddiffpairs[0]), np.cos(rnddiffpairs[1]-rnddiffpairs[0])))

	allrnddiffpeaks[s] = rnddiffpeaks
	allrnddiffsess[s] = rnddiffsess

allrnddiffpeaks = pd.concat(allrnddiffpeaks, 1)
allrnddiffsess = pd.concat(allrnddiffsess, 1)

###############################################################################################
# FIGURES
###############################################################################################

bins = np.linspace(0, 180, 16)

figure()
subplot(221)
# a, b = np.histogram(np.rad2deg(allrnddiffpeaks.values.flatten()), bins, density = True)
# plot(b[0:-1], a)
xy = [np.rad2deg(diffpeaks.values), np.rad2deg(allrnddiffpeaks.values.flatten())]
hist(xy, bins, density = True, label = ['', 'random'], histtype='step', linewidth = 5)
legend()
xticks([0, 30, 60, 90, 120, 150, 180])
title("Diff of peaks")

subplot(222)
xy = [np.rad2deg(diffsess.values), np.rad2deg(allrnddiffsess.values.flatten())]
hist(xy, bins, density = True, label = ['', 'random'], histtype='step', linewidth = 5)
legend()
xticks([0, 30, 60, 90, 120, 150, 180])
title("Diff of diff of pairs")
bins = np.linspace(0, 180, 120)

subplot(223)
xy = [np.rad2deg(diffpeaks.values), np.rad2deg(allrnddiffpeaks.values.flatten())]
xy2 = []
for x in xy:
	tmp, _ = np.histogram(x, bins)	
	tmp = tmp / tmp.sum()
	tmp2 = pd.Series(data = np.hstack((tmp, tmp, tmp)))
	tmp2 = tmp2.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=5)
	tmp3 = tmp2.values.reshape(3,len(tmp))
	xy2.append(pd.Series(index = bins[0:-1]+np.diff(bins)/2, data = tmp3[1]))
xy2 = pd.concat(xy2, 1)
plot(xy2)
xticks([0, 30, 60, 90, 120, 150, 180])

subplot(224)
xy = [np.rad2deg(diffsess.values), np.rad2deg(allrnddiffsess.values.flatten())]
xy2 = []
for x in xy:
	tmp, _ = np.histogram(x, bins)	
	tmp = tmp / tmp.sum()
	tmp2 = pd.Series(data = np.hstack((tmp, tmp, tmp)))
	tmp2 = tmp2.rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=5)
	tmp3 = tmp2.values.reshape(3,len(tmp))
	xy2.append(pd.Series(index = bins[0:-1]+np.diff(bins)/2, data = tmp3[1]))
xy2 = pd.concat(xy2, 1)
plot(xy2)
xticks([0, 30, 60, 90, 120, 150, 180])


###############################################################################################
# 
###############################################################################################
