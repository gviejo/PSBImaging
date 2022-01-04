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
# from pycircstat.descriptive import mean as circmean
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


##################################################################################################
# DIFFERENT ENVIRONMENTS FOR POSTSUB
##################################################################################################
data_directory = '/mnt/DataRAID/MINISCOPE'
#data_directory = '/media/guillaume/Elements'
struct = pd.DataFrame(index = ['A0634', 'A0642'], 
	data = [['psb', (166, 136)],
			['psb', (201, 211)]
			],
	columns = ['struct', 'dims'])

############################################################
# ANIMALS INFO
############################################################
infos = loadInfos(data_directory, '/home/guillaume/PSBImaging/python/', struct.index)
allreg = {}
allstd = {}

for a in infos.keys():
	############################################################
	# LOADING DATA
	############################################################
	SF, TC, PF, allinfo, positions, DFFS, Cs = loadDatas(infos[a]['paths'], struct.loc[a,'dims'])
	cellreg, scores = loadCellReg(os.path.join(data_directory, a[0:3] + '00', a))

	############################################################
	# SELECTING DATA
	############################################################
	sessions = infos[a].index.values
	n_sessions_detected = np.sum(cellreg!=-1, 1)
	
	# Detected in at least 1 session
	tokeep = np.where(n_sessions_detected > 5)[0]

	# Good Cell reg scores 
	tokeep = tokeep[scores[tokeep]>0.8]

	# Selecting neurons with stable tuning curves
	allst = {}
	for i in tokeep:
		allst[i] = pd.Series(index = np.arange(cellreg.shape[1]), dtype = np.float32)
		for j in np.where(cellreg[i]!=-1)[0]:
			allst[i][j] = allinfo[list(allinfo.keys())[j]]['halfcorr'].loc[cellreg[i,j]]

	allst = pd.concat(allst, 1).T
	allst[allst<0.2] = np.nan
	tokeep = allst[allst.notna().any(1)].index.values
		

	# Selecting HD cells
	# alltc = {}
	# for i in tokeep: # neuron index
	# 	alltc[i] = pd.DataFrame(columns = sessions)
	# 	for j in np.where(cellreg[i]!=-1)[0]:
	# 		alltc[i][sessions[j]] = TC[list(TC.keys())[j]][cellreg[i,j]]

	# std = findSinglePeakHDCell(alltc, sessions)	
	# tokeep = std.index[std.mean(1) < std.mean(1).quantile(0.8)]

	print(len(tokeep))

	allreg[a] = cellreg[tokeep]

	allstd[a] = std

####################################################
# CLUSTERING BASED ON CELL REG	
####################################################
order = ['Circular', 'Square', '8-arm maze', 'Open field']

clus = {}
for a in allreg.keys():
	tmp = allreg[a].copy()
	tmp[tmp > -1] = 1
	# keeping sessions with less than 50 % of -1
	p = np.sum(tmp==-1, 0)/len(tmp)
	good_sessions = np.where(p < 0.5)[0]
	imap = UMAP(n_neighbors = 6, min_dist = 0.0).fit_transform(tmp[:,good_sessions].T)
	#imap = TSNE(n_components = 2, perplexity = 1).fit_transform(tmp.T)
	#imap = Isomap(n_components = 2, n_neighbors = 20).fit_transform(tmp.T)
	classe = np.array([list(order).index(e) for e in infos[a]['Rig'].values])
	data = pd.DataFrame(index = classe[good_sessions], data = imap)
	clus[a] = data


#####################################################
# COMMON CELLS
#####################################################
allcorr = {}
for a in allreg.keys():
	classe = np.array([list(order).index(e) for e in infos[a]['Rig'].values])
	cellreg = allreg[a]
	tmp = cellreg.copy()
	tmp[tmp > -1] = 1
	corr = np.zeros((cellreg.shape[1], cellreg.shape[1]))
	for i,j in combinations(np.arange(tmp.shape[1]), 2):
		corr[i,j] = np.sum(np.prod(tmp[:,[i,j]], 1)==1) / tmp.shape[0]
		corr[j,i] = corr[i,j]
	allcorr[a] = pd.DataFrame(index = classe, columns = classe, data = corr)



figure()
gs = GridSpec(1,2)

############### 
# DIFFERENT ENVS
grps = struct.groupby('struct').groups

for j, a in enumerate(grps['psb']):
	subplot(gs[0,j])
	color = iter(cm.rainbow(np.linspace(0, 1, len(order))))	
	classe = clus[a].index.values
	imap = clus[a].values		
	for k in range(4):
		scatter(imap[classe==k,0], imap[classe==k,1], color = next(color), label = order[k])
	legend()		
	title(a)

show()


sys.exit()



##################################################################################################
# SAME ENVIRONMENTS FOR POSTSUB
##################################################################################################
############################################################
# ANIMAL INFO
############################################################
fbasename = 'A0634'
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_same_env_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
paths = [os.path.join(data_directory, fbasename[0:3] + '00', fbasename, fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
sessions = [fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '') for i in info.index]
info['paths'] = paths
info['sessions'] = sessions
info = info.set_index('sessions')

if fbasename == 'A0634':
	dims = (304,304)
elif fbasename == 'A0642':
	dims = (201, 211)

############################################################
# LOADING DATA
############################################################
SF, TC, PF, allinfo, positions, DFFS, Cs = loadDatas(paths, dims)
cellreg, scores = loadCellReg(os.path.join(data_directory, fbasename[0:3] + '00', fbasename), 'CellRegSameEnvs')

####################################################
# CLUSTERING BASED ON CELL REG	
####################################################
tmp = cellreg.copy()
tmp[tmp > -1] = 1
imap2 = UMAP(n_neighbors = 5, min_dist = 0.1, low_memory=True).fit_transform(tmp.T)

#####################################################
# COMMON CELLS
#####################################################
tmp = cellreg.copy()
tmp[tmp > -1] = 1
corr = np.zeros((tmp.shape[1],tmp.shape[1]))

for i,j in combinations(np.arange(tmp.shape[1]), 2):
	corr[i,j] = np.sum(np.prod(tmp[:,[i,j]], 1)==1) / tmp.shape[0]
	corr[j,i] = corr[i,j]



##################################################################################
# FIGURES
##################################################################################

figure(figsize = (17, 10))
gs = GridSpec(3,3)


############### 
# DIFFERENT ENVS
grps = struct.groupby('struct').groups

for j, a in enumerate(grps['psb']):
	subplot(gs[0,j])
	color = iter(cm.rainbow(np.linspace(0, 1, len(order))))	
	classe = clus[a].index.values
	imap = clus[a].values		
	for k in range(4):
		scatter(imap[classe==k,0], imap[classe==k,1], color = next(color), label = order[k])
	legend()		
	title(a)
###############
# SAME ENVS
subplot(gs[0,2])
scatter(imap2[:,0], imap2[:,1], c = np.arange(0, len(imap2)))
title('A0634 same env 13 days')

###############
#CORR
vmin = np.min([allcorr[a].values[np.triu_indices_from(allcorr[a], 1)].min() for a in allcorr])
vmax = np.max([allcorr[a].values[np.triu_indices_from(allcorr[a], 1)].max() for a in allcorr])
vmin = np.min([vmin, corr[np.triu_indices_from(corr, 1)].min()])
vmax = np.min([vmax, corr[np.triu_indices_from(corr, 1)].max()])


for i, a in enumerate(allcorr.keys()):
	subplot(gs[1,i])
	imshow(allcorr[a], vmin = vmin, vmax = vmax)
	for s in np.unique(infos[a]['Rig']):
		axvline(np.where(infos[a]['Rig'] == s)[0][0]-0.5 , color = 'red')
		axvline(np.where(infos[a]['Rig'] == s)[0][-1]+0.5, color = 'red')
		axhline(np.where(infos[a]['Rig'] == s)[0][0]-0.5 , color = 'red')
		axhline(np.where(infos[a]['Rig'] == s)[0][-1]+0.5, color = 'red')

subplot(gs[1,-1])
imshow(corr, vmin = vmin, vmax = vmax)

for i, a in enumerate(allcorr.keys()):
	subplot(gs[2,i])
	title(a)
	envs = infos[a].groupby('Rig').groups		
	sessions = infos[a].index.values
	xcorr = pd.DataFrame(index = sessions, columns = sessions, data = allcorr[a].values)
	tmp = {}
	for e in envs.keys():
		tmp[e] = np.array([xcorr.loc[m,n] for m, n in combinations(envs[e],2)])
	inter = set(combinations(sessions,2)) - set(sum([list(combinations(envs[e],2)) for e in envs.keys()]))
	tmp['inter'] = np.array([xcorr.loc[m,n] for m, n in inter])

	violinplot(tmp.values())
	
	m = [tmp[e].mean() for e in tmp.keys()]
	v = [tmp[e].std() for e in tmp.keys()]

	errorbar(range(1,len(m)+1),m,v, marker = 'o', linestyle = 'None')

	xticks(range(1,len(tmp)+1), tmp.keys(), rotation = 20)		


subplot(gs[2,-1])
for i in range(corr.shape[1]):
	plot(np.arange(i+1, corr.shape[1]), corr[i,i+1:])



savefig('../figures/figure_RIG_decoding_all.pdf', dpi = 200, bbox_inches = 'tight')

#show()

