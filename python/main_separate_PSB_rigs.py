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
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
from umap import UMAP

data_directory = '/mnt/DataRAID/MINISCOPE'
#data_directory = '/media/guillaume/Elements'

############################################################
# ANIMAL INFO
############################################################
fbasename = 'A0642'
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

############################################################
# LOADING DATA
############################################################
SF, TC, PF, allinfo, positions, DFFS, Cs = loadDatas(paths, dims)


cellreg, scores = loadCellReg(os.path.join(data_directory, fbasename[0:3] + '00', fbasename))

tolook = np.arange(len(sessions))
############################################################
# CORRELATING SUM OF DFFS
############################################################
n_sessions_detected = np.sum(cellreg != -1, 1)
tokeep = np.where(n_sessions_detected > 5)[0]
tokeep = tokeep[scores[tokeep]>0.8]

# Selecting neurons with stable tuning curves
allst = {}
for i in tokeep: # neuron index
	allst[i] = pd.Series(index = tolook, dtype = np.float32)
	for j in np.where(cellreg[i]!=-1)[0]:
		allst[i][j] = allinfo[tolook[j]]['halfcorr'].loc[cellreg[i,j]]

allst = pd.concat(allst, 1).T
allst[allst<0.7] = np.nan
tokeep = allst[allst.notna().any(1)].index.values


sumdff = np.ones_like(cellreg)*np.nan
for i,s in enumerate(sessions):
	idx = cellreg[:,i][cellreg[:,i] != -1]
	tmp = Cs[i][idx]
	tmp2 = []
	for j,n in enumerate(tmp.columns):
		idx, _ = scipy.signal.find_peaks(tmp[n], height=tmp[n].quantile(0.5))
		sumdff[j,i] = len(idx)
	
	#tmp = tmp / tmp.max()		
	#tmp2 = tmp.sum(0)
	#tmp2 = tmp2/tmp2.sum()
#	sumdff[:,i][cellreg[:,i] != -1] = tmp2.values

sumdff = sumdff[tokeep]

sumdff = sumdff/np.nansum(sumdff,0)

tmp = np.nanmean(sumdff, 0)
for i in range(sumdff.shape[1]):
 	idx = np.isnan(sumdff[:,i])
 	for j in np.where(idx)[0]:
 		sumdff[j,i] = np.nanmean(sumdff[:,i])

sumdff[np.isnan(sumdff)] = 0

#imap = TSNE(n_components = 2, perplexity = 5).fit_transform(sumdff.T)
#imap = PCA(n_components = 2).fit_transform(sumdff.T)
imap = UMAP(n_neighbors = 5, min_dist = 0.0001, low_memory=True).fit_transform(sumdff.T)

classe = np.array([list(np.unique(info['Rig'].iloc[tolook])).index(e) for e in info['Rig'].iloc[tolook].values])
color = iter(cm.rainbow(np.linspace(0, 1, 4)))
figure()
for i in range(4):
	scatter(imap[classe==i,0], imap[classe==i,1], c = next(color), label = np.unique(info['Rig'].iloc[tolook])[i])
legend()
show()

# shuffling
tmp = sumdff.copy()
shuff = []
for i in range(tmp.shape[1]):
	tmp2 = tmp[:,i]
	np.random.shuffle(tmp2)
	shuff.append(tmp2)
shuff = np.array(shuff)

imap = UMAP(n_neighbors = 5, min_dist = 0.0001, low_memory=True).fit_transform(shuff.T)
classe = np.array([list(np.unique(info['Rig'].iloc[tolook])).index(e) for e in info['Rig'].iloc[tolook].values])
color = iter(cm.rainbow(np.linspace(0, 1, 4)))
figure()
for i in range(4):
	scatter(imap[classe==i,0], imap[classe==i,1], c = next(color), label = np.unique(info['Rig'].iloc[tolook])[i])
legend()
show()





corr = np.zeros((len(sessions), len(sessions)))

for i,j in combinations(range(len(sessions)), 2):
	tmp = sumdff[:,[i,j]]
	tmp = tmp[~np.isnan(tmp).any(1)]
	corr[i,j] = np.corrcoef(tmp.T)[0,1]
	corr[j,i] = np.corrcoef(tmp.T)[0,1]

figure()
imshow(corr)
for s in np.unique(info['Rig']):
	axvline(np.where(info['Rig'] == s)[0][0]-0.5 , color = 'red')
	axvline(np.where(info['Rig'] == s)[0][-1]+0.5, color = 'red')
	axhline(np.where(info['Rig'] == s)[0][0]-0.5 , color = 'red')
	axhline(np.where(info['Rig'] == s)[0][-1]+0.5, color = 'red')
show()

#########################################################
# SUM OF TUNING CURVES
#########################################################

sumtc = np.ones_like(cellreg)*np.nan
for i,s in enumerate(tolook):
	idx = cellreg[:,i][cellreg[:,i] != -1]
	tmp = TC[i][idx]
	#tmp = tmp / tmp.max()
	#tmp2 = tmp.sum(0)
	#tmp2 = tmp2/tmp2.sum()
	tmp2 = tmp.sum(0)
	sumtc[:,i][cellreg[:,i] != -1] = tmp2.values


corr = np.zeros((len(sessions), len(sessions)))

for i,j in combinations(range(len(sessions)), 2):
	tmp = sumtc[:,[i,j]]
	tmp = tmp[~np.isnan(tmp).any(1)]
	corr[i,j] = np.corrcoef(tmp.T)[0,1]
	corr[j,i] = np.corrcoef(tmp.T)[0,1]

figure()
imshow(corr)
for s in np.unique(info['Rig']):
	axvline(np.where(info['Rig'] == s)[0][0]-0.5 , color = 'red')
	axvline(np.where(info['Rig'] == s)[0][-1]+0.5, color = 'red')
	axhline(np.where(info['Rig'] == s)[0][0]-0.5 , color = 'red')
	axhline(np.where(info['Rig'] == s)[0][-1]+0.5, color = 'red')
show()

############################################################
# SELECTING DATA
############################################################
# Sessions with most cellreg
tolook = np.sort(np.argsort(np.sum(cellreg != -1, 0))[4:])
#tolook = np.sort(np.argsort(np.sum(cellreg != -1, 0)))

cellreg2 = cellreg[:,tolook]
#tolook = np.arange(len(sessions))

n_sessions_detected = np.sum(cellreg2 != -1, 1)

# Detected in most sessions
tokeep = np.where(n_sessions_detected == len(tolook))[0]

# Good Cell reg scores 
tokeep = tokeep[scores[tokeep]>0.8]

# Selecting neurons with stable tuning curves
allst = {}
for i in tokeep: # neuron index
	allst[i] = pd.Series(index = tolook, dtype = np.float32)
	for j in np.where(cellreg2[i]!=-1)[0]:
		allst[i][j] = allinfo[tolook[j]]['halfcorr'].loc[cellreg2[i,j]]

allst = pd.concat(allst, 1).T
allst[allst<0.5] = np.nan
tokeep = allst[allst.notna().any(1)].index.values

# Selecting HD cells
# alltc = {}
# for i in tokeep: # neuron index
# 	alltc[i] = pd.DataFrame(columns = tolook)	
# 	for j in np.where(cellreg2[i]!=-1)[0]:
# 		alltc[i][j] = TC[list(TC.keys())[j]][cellreg2[i,j]]

# std = findSinglePeakHDCell(alltc, sessions)
# #std[std>0.7] = np.nan
# tokeep = std.dropna(0).index.values


#########################################################
# SUM OF CALCIUM TRANSIENTS
#########################################################
sumdff = []

for i , s in enumerate(tolook):	
	idx_sessions = cellreg2[tokeep][:,i]
	tmp = Cs[s][idx_sessions].values
	tmp = tmp / tmp.max()	
	tmp2 = tmp/tmp.sum()
	tmp2 = tmp.sum(0)
	tmp2 = tmp2/tmp2.sum()
	sumdff.append(tmp2)

sumdff = pd.DataFrame(index = tokeep, data = np.array(sumdff).T)

#imap = PCA(n_components = 2).fit_transform(sumdff.T)
#imap = UMAP(n_neighbors = 15, min_dist = 0.01, low_memory=True).fit_transform(sumdff.T)
imap = TSNE(n_components = 2, perplexity = 5).fit_transform(sumdff.T)


clrs = [list(np.unique(info['Rig'].iloc[tolook])).index(e) for e in info['Rig'].iloc[tolook].values]


figure()
scatter(imap[:,0], imap[:,1], c = clrs, cmap = 'jet')
show()