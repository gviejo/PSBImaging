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
	tokeep 			= si[(si>thr_si).values].index.values
	SI[i] 			= si
	SF[i] 			= A
	TC[i] 			= tuningcurve
	HD[i] 			= tokeep
	PK[i] 			= peaks
	RL[i] 			= stat



########################################################################################
# LOAD CELL REG SESSIONS PAIRS
########################################################################################
path = '/mnt/DataRAID/MINISCOPE/A0600/A0634/CellRegPairs'
files = os.listdir(path)
cellreg_file = np.sort([f for f in files if 'cellRegistered' in f])

alltc = {0:[], 1:[]}
allsi = {0:[], 1:[]}
allpk = {0:[], 1:[]}
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
		tc.columns = names
		si.index = names
		pk.index = names
		alltc[j].append(tc)
		allsi[j].append(si)
		allpk[j].append(pk)

for j in range(2):
	alltc[j] = pd.concat(alltc[j], 1)
	allsi[j] = pd.concat(allsi[j])
	allpk[j] = pd.concat(allpk[j])

allsi = pd.concat(allsi, 1)
allpk = pd.concat(allpk, 1)

tokeep = allsi[(allsi>thr_si).all(1)].index.values


###############################################################################################
# TUNING CURVES PER SESSIONS
###############################################################################################

for i in TC.keys():
	tcurves = TC[i]
	# tokeep = SI[i][(SI[i]>thr_si).values].index.values
	tokeep = RL[i][(RL[i]>thr_rl).values].index.values
	figure()
	for j in tcurves.columns:
		subplot(int(np.ceil(np.sqrt(tcurves.shape[1]))),int(np.ceil(np.sqrt(tcurves.shape[1]))),j+1, projection='polar')
		plot(tcurves[j])
		if len(tokeep):
			if j in tokeep:				
				fill_between(tcurves[j].index.values, np.zeros(len(tcurves[j])), tcurves[j].values, color = 'grey')
		xticks([])
		yticks([])
		#title(np.round(SI[i].loc[j][0], 2))
		title('z='+str(np.round(RL[i].loc[j,'z'], 2)))

###############################################################################################
# TUNING CURVES ACROSS SESSIONS
###############################################################################################

figure()
index = alltc[0].columns
for i, n in enumerate(index):
	subplot(int(np.ceil(np.sqrt(len(index)))),int(np.ceil(np.sqrt(len(index)))),i+1, projection='polar')
	for j in alltc.keys():
		plot(alltc[j][n])
	xticks([])
	yticks([])