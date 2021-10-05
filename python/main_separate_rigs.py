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


data_directory = '/mnt/DataRAID/MINISCOPE'
#data_directory = '/media/guillaume/Elements'
struct = pd.DataFrame(index = ['A0634', 'A0642', 'A6509', 'A6510'], 
	data = [['psb', (166, 136)],
			['psb', (201, 211)],
			['rsp', (202, 192)],
			['rsp', (192, 251)]],			
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
	tokeep = np.where(n_sessions_detected > 3)[0]

	# Good Cell reg scores 
	tokeep = tokeep[scores[tokeep]>0.8]

	# Selecting neurons with stable tuning curves
	allst = {}
	for i in tokeep:
		allst[i] = pd.Series(index = np.arange(cellreg.shape[1]), dtype = np.float32)
		for j in np.where(cellreg[i]!=-1)[0]:
			allst[i][j] = allinfo[list(allinfo.keys())[j]]['halfcorr'].loc[cellreg[i,j]]

	allst = pd.concat(allst, 1).T
	allst[allst<0.1] = np.nan
	tokeep = allst[allst.notna().any(1)].index.values
	
	# Selecting HD cells
	alltc = {}
	for i in tokeep: # neuron index
		alltc[i] = pd.DataFrame(columns = sessions)
		for j in np.where(cellreg[i]!=-1)[0]:
			alltc[i][sessions[j]] = TC[list(TC.keys())[j]][cellreg[i,j]]

	std = findSinglePeakHDCell(alltc, sessions)
	#std[std>0.7] = np.nan
	#tokeep = std.dropna(0).index.values

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
	imap = UMAP(n_neighbors = 5, min_dist = 0.1, low_memory=True).fit_transform(tmp.T)
	classe = np.array([list(order).index(e) for e in infos[a]['Rig'].values])
	data = pd.DataFrame(index = classe, data = imap)
	clus[a] = data


figure()
gs = GridSpec(2,2)

grps = struct.groupby('struct').groups

for i, g in enumerate(grps.keys()):
	for j, a in enumerate(grps[g]):
		subplot(gs[i,j])
		color = iter(cm.rainbow(np.linspace(0, 1, len(order))))	
		classe = clus[a].index.values
		imap = clus[a].values
		size = np.arange(0, np.sum(classe==k))*10
		for k in range(4):
			scatter(imap[classe==k,0], imap[classe==k,1], color = next(color), label = order[k])
		legend()		
		title(a)

show()		

##################################################
# SHUFFLING
##################################################

order = ['Circular', 'Square', '8-arm maze', 'Open field']

clus = {}
for a in allreg.keys():
	tmp = allreg[a].copy()
	tmp[tmp > -1] = 1

	# tmp3 = []
	# for i in range(tmp.shape[1]):
	# 	tmp2 = tmp[:,i].copy()
	# 	np.random.shuffle(tmp2)
	# 	tmp3.append(tmp2)
	# tmp3 = np.array(tmp3)

	tmp3 = []
	for i in range(tmp.shape[0]):
		tmp2 = tmp[i].copy()
		np.random.shuffle(tmp2)
		tmp2 = np.roll(tmp2, np.random.randint(100))
		tmp3.append(tmp2)
	tmp3 = np.array(tmp3)

	# tmp3 = np.copy(tmp).T
	# np.random.shuffle(tmp3)
	# tmp3 = tmp3.T
	imap = UMAP(n_neighbors = 10, min_dist = 0.1, low_memory=True).fit_transform(tmp3.T)
	classe = np.array([list(order).index(e) for e in infos[a]['Rig'].values])
	data = pd.DataFrame(index = classe, data = imap)
	clus[a] = data


figure()
gs = GridSpec(2,2)

grps = struct.groupby('struct').groups

for i, g in enumerate(grps.keys()):
	for j, a in enumerate(grps[g]):
		subplot(gs[i,j])
		color = iter(cm.rainbow(np.linspace(0, 1, len(order))))	
		classe = clus[a].index.values
		imap = clus[a].values
		for k in range(4):
			scatter(imap[classe==k,0], imap[classe==k,1], color = next(color), label = order[k])
		legend()		
		title(a)



##################################################
# CLUSTERING BASED ON STD
##################################################

order = ['Circular', 'Square', '8-arm maze', 'Open field']

clus = {}
for a in allreg.keys():
	std = allstd[a]

	tmp = std.mean(1,skipna=True)
	# grps = [np.where(tmp>tmp.quantile(0.5))[0],
	# 		np.where(tmp<tmp.quantile(0.5))[0]]
	grps = [np.where(tmp>1.0)[0],
			np.where(tmp<1.0)[0]]
	tmp = allreg[a].copy()
	tmp[tmp > -1] = 1

	clus[a] = {}
	for i in range(2):
		imap = UMAP(n_neighbors = 5, min_dist = 0.1, low_memory=True).fit_transform(tmp[grps[i]].T)
		classe = np.array([list(order).index(e) for e in infos[a]['Rig'].values])
		data = pd.DataFrame(index = classe, data = imap)
		clus[a][i] = data

figure()
gs = GridSpec(2,2)

grps = struct.groupby('struct').groups

for i, g in enumerate(grps.keys()):
	for j, a in enumerate(grps[g]):
		gs2 = GridSpecFromSubplotSpec(1,2,gs[i,j])
		#subplot(gs[i,j])
		for n in range(2):
			subplot(gs2[0,n])
			color = iter(cm.rainbow(np.linspace(0, 1, len(order))))	
			classe = clus[a][n].index.values
			imap = clus[a][n].values
			for k in range(4):
				scatter(imap[classe==k,0], imap[classe==k,1], color = next(color), label = order[k])
			legend()		
			title(a)



#####################################################
# COMMON CELLS
#####################################################
for a in allreg.keys():
	classe = np.array([list(order).index(e) for e in infos[a]['Rig'].values])
	cellreg = allreg[a]
	corr = 	np.zeros((4,4))
	for i, j in combinations(range(4),2):
		tmp = cellreg[:,np.logical_or(classe==i, classe==j)]
		n_sessions_detected = np.sum(tmp>-1,1)
		corr[i,j] = np.sum(n_sessions_detected == np.max(n_sessions_detected))/len(n_sessions_detected)
		corr[j,i] = corr[i,j]

figure()
subplot(121)
imshow(corr)
subplot(122)
plot(range(3), corr[0,1:])
plot(range(1,3),corr[1,2:])
show()