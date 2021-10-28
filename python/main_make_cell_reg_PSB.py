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
#import cv2
from matplotlib.gridspec import GridSpec
from itertools import product, combinations
from scipy.stats import norm
from scipy.ndimage.filters import gaussian_filter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


data_directory = '/mnt/DataRAID/MINISCOPE'
#data_directory = '/media/guillaume/Elements'

############################################################
# ANIMAL INFO
############################################################
fbasename = 'A0643'
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
paths = [os.path.join(data_directory, fbasename[0:3] + '00', fbasename, fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
sessions = [fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '') for i in info.index]
info['paths'] = paths
info['sessions'] = sessions
info = info.set_index('sessions')

if fbasename == 'A0634':
	dims = (166, 136)
elif fbasename == 'A0642':
	dims = (201, 211)
elif fbasename == 'A0643':
	dims = (186,186)

############################################################
# LOADING DATA
############################################################
SF, TC, PF, allinfo, positions, DFFS, Cs = loadDatas(paths, dims)


cellreg, scores = loadCellReg(os.path.join(data_directory, fbasename[0:3] + '00', fbasename))


############################################################
# SELECTING DATA
############################################################
n_sessions_detected = np.sum(cellreg!=-1, 1)

# Detected in most sessions
tokeep = np.where(n_sessions_detected >= 5)[0]

# Good Cell reg scores 
tokeep = tokeep[scores[tokeep]>0.5]

# Selecting neurons with stable tuning curves
allst = {}
for i in tokeep:
	allst[i] = pd.Series(index = np.arange(cellreg.shape[1]), dtype = np.float32)
	for j in np.where(cellreg[i]!=-1)[0]:
		allst[i][j] = allinfo[list(allinfo.keys())[j]]['halfcorr'].loc[cellreg[i,j]]

allst = pd.concat(allst, 1).T
allst[allst<0.4] = np.nan
tokeep = allst[allst.notna().any(1)].index.values




alltc = {}
allpf = {}
allpk = pd.DataFrame(index = tokeep, columns = sessions, dtype = np.float32)
for i in tokeep:
	alltc[i] = pd.DataFrame(columns = sessions)
	allpf[i] = np.zeros((len(sessions), PF[0].shape[1], PF[0].shape[2],))	
	for j in np.where(cellreg[i]!=-1)[0]:
		alltc[i][sessions[j]] = TC[list(TC.keys())[j]][cellreg[i,j]]
		allpf[i][j] = PF[list(TC.keys())[j]][cellreg[i,j]]
		allpk.loc[i,sessions[j]] = allinfo[j].loc[cellreg[i,j],'peaks']


####################################################################################
# DIFF PEAKS + SESSIONS
####################################################################################
diffpeaks = computePeaksAngularDifference(alltc, sessions = sessions)
diffsess = computePairwiseAngularDifference(alltc, sessions = sessions)



####################################################################################
# RANDOMIZING 
####################################################################################
rnddiffsess = []
for k in range(2):
	print(k)
	rndcellreg = np.copy(cellreg[list(alltc.keys())])
	for t in range(rndcellreg.shape[1]):
		np.random.shuffle(rndcellreg[:,t])
	rndtc = {}
	for i in range(len(rndcellreg)):
		rndtc[i] = pd.DataFrame(columns = sessions)
		for j in np.where(rndcellreg[i]!=-1)[0]:
			rndtc[i][sessions[j]] = TC[list(TC.keys())[j]][rndcellreg[i,j]]

	rnddiffsess.append(computePairwiseAngularDifference(rndtc, sessions = sessions))

rnddiffsess = pd.concat(rnddiffsess)



#####################################################################################
# PLOT SUMMARY PLOT
#####################################################################################
envs = info.loc[sessions].groupby('Rig').groups

order = ['Circular', 'Square', '8-arm maze', 'Open field']

sessions = list(sessions)

ref = sessions[np.sum(cellreg[tokeep] == -1, 0).argmin()]# reference sessions

norder = allpk[ref].dropna().sort_values().index.values

figure(figsize = (15, 10))
gs = GridSpec(len(order), 1, top = 0.95, bottom = 0.05, left = 0.03, right = 0.96)

for i, o in enumerate(order):
	n_envs = len(envs[o])
	gs2 = GridSpecFromSubplotSpec(2,n_envs, gs[i,:])

	for j, s in enumerate(envs[o]):
		gs3 = GridSpecFromSubplotSpec(1,2,gs2[0,j])
		# plot position of each exploration		
		ax = subplot(gs3[0,0], aspect = 'equal')
		if s!=ref: noaxis(ax)
		plot(positions[sessions.index(s)]['x'], positions[sessions.index(s)]['z'])
		xticks([])
		yticks([])

		# plot spatial footprints colored by preferessed direction	
		ax = subplot(gs3[0,1], aspect = 'equal')
		A = SF[sessions.index(s)]
		cA = getColoredFootprints(A, allinfo[sessions.index(s)]['peaks'], 4)
		imshow(cA, cmap = 'jet')
		xticks([])
		yticks([])

	# plot matrix of tc
	for j, s in enumerate(envs[o][0:n_envs]):		
		tc = np.array([alltc[n][s].values for k, n in enumerate(norder) if ~np.isnan(alltc[n][s].values[0])])
		#tc = (tc.T/tc.max(1)).T
		tc2 = gaussian_filter(tc, 2)	
		if len(tc2):	
			ax = subplot(gs2[1,j], aspect = 'equal')
			#noaxis(ax)
			imshow(tc2, cmap = 'jet', aspect = 'auto')
			xticks([])
		
savefig('../figures/figure_psb_'+fbasename+'_1.pdf', format = 'pdf', dpi = 200, bbox_inches = 'tight')



####################################################################################
# BLURRED SPATIAL FOOTPRINTS
####################################################################################
figure()
gs = GridSpec(len(order), 1, top = 0.95, bottom = 0.05, left = 0.03, right = 0.96)

for i, o in enumerate(order):
	n_envs = len(envs[o])
	gs2 = GridSpecFromSubplotSpec(1,n_envs, gs[i,:])

	for j, s in enumerate(envs[o]):
		# plot spatial footprints colored by preferessed direction	
		ax = subplot(gs2[0,j], aspect = 'equal')
		A = SF[sessions.index(s)]		
		dims = A.shape[1:]
		peaks = allinfo[sessions.index(s)]['peaks']
		alpha = np.zeros_like(A)*np.nan
		thr = 2
		for i in range(len(A)):
			alpha[i][A[i] > 1] = peaks[i]

		meanalpha = np.arctan2(np.nanmean(np.sin(alpha), 0), np.nanmean(np.cos(alpha), 0))
		meanalpha[meanalpha<0] += 2*np.pi

		H = meanalpha/(2*np.pi)
		HSV = np.dstack((H, np.ones_like(H), np.ones_like(H)))
		colorA = hsv_to_rgb(HSV)

		imshow(colorA)
		xticks([])
		yticks([])



####################################################################################
# PLOT TEMPORAL PAIRWISE ANGULAR DIFFERENCE
####################################################################################

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



show()









