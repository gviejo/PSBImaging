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
fbasename = 'A6509'
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
paths = [os.path.join(data_directory, fbasename[0:3] + '00', fbasename, fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
sessions = [fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '') for i in info.index]
info['paths'] = paths
info['sessions'] = sessions
info = info.set_index('sessions')

if fbasename == 'A6509':
	dims = (202,192)
elif fbasename == 'A6510':
	dims = (192,251)


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
		ax = subplot(gs2[1,j], aspect = 'equal')
		#noaxis(ax)
		imshow(tc2, cmap = 'jet')
		xticks([])
		
savefig('../figures/figure_rsp_'+fbasename+'_1.pdf', format = 'pdf', dpi = 200, bbox_inches = 'tight')



####################################################################################
# TUNING CURVES CORRELATION
####################################################################################
pairs = list(combinations(sessions, 2))
tc_neuron = {}
for s in sessions:
	tc_neuron[s] = []
	for n in alltc.keys():
		tc_neuron[s].append(alltc[n][s])
	tc_neuron[s] = pd.concat(tc_neuron[s], 1)
	tc_neuron[s].columns = tokeep

corr_tc_sess = {}

for p in pairs:
	tmp1 = tc_neuron[p[0]].dropna(1)
	tmp2 = tc_neuron[p[1]].dropna(1)
	tmp3 = np.intersect1d(tmp1.columns, tmp2.columns)
	c = np.corrcoef(tmp1[tmp3].values, tmp2[tmp3].values)
	corr_tc_sess[p] = c[0:len(tmp1), len(tmp1):]

figure(figsize = (15, 10))
gs = GridSpec(len(sessions), len(sessions), top = 0.95, bottom = 0.05, left = 0.03, right = 0.96)

for i, p in zip(combinations(range(len(sessions)),2), pairs):
	subplot(gs[i[0],i[1]])
	imshow(corr_tc_sess[p])


