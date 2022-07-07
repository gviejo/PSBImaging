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
from matplotlib import colors

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


for a in ['A0634', 'A0642']:
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

inter_dist = {}
intra_dist = {}

inter_dist_shuf = {}
intra_dist_shuf = {}

ratios = {}

for a in allreg.keys():
	tmp = allreg[a].copy()
	tmp[tmp > -1] = 1
	# keeping sessions with less than 50 % of -1
	p = np.sum(tmp==-1, 0)/len(tmp)
	good_sessions = np.where(p < 0.5)[0]
	imap = UMAP(n_neighbors = 5, min_dist = 0.1).fit_transform(tmp[:,good_sessions].T)
	imap = (imap - imap.mean(0))/imap.std(0)
	classe = np.array([list(order).index(e) for e in infos[a]['Rig'].values])
	data = pd.DataFrame(index = classe[good_sessions], data = imap)
	
	

	intra_dist[a] = []

	for i in np.unique(data.index):
		tmp2 = data.loc[i].values
		dist = np.sqrt(np.power(tmp2[:,[0]] - tmp2[:,0], 2) + np.power(tmp2[:,[1]] - tmp2[:,1], 2))
		intra_dist[a].append(dist[np.triu_indices_from(dist, 1)])
	intra_dist[a] = np.hstack(intra_dist[a])

	inter_dist[a] = []
	for i, j in combinations(range(4),2):
		tmp1 = data.loc[i].values
		tmp2 = data.loc[j].values	

		dist = np.sqrt(np.power(tmp1[:,[0]] - tmp2[:,0], 2) + np.power(tmp1[:,[1]] - tmp2[:,1], 2))
		inter_dist[a].append(dist.flatten())
	inter_dist[a] = np.hstack(inter_dist[a])

	ratios[a] = {}
	ratios[a]['true'] = intra_dist[a].mean()/inter_dist[a].mean()
	ratios[a]['shuf'] = []
		

	###################################
	# SHUFFLING
	tmp = allreg[a].copy()
	tmp[tmp > -1] = 1
	# keeping sessions with less than 50 % of -1
	p = np.sum(tmp==-1, 0)/len(tmp)
	good_sessions = np.where(p < 0.5)[0]
	tmp = tmp[:,good_sessions]
	
	for k in range(500):		
		print(a, k)
		tmp2 = tmp.copy()
		for l in range(tmp.shape[1]):
			tmp2[:,l] = np.random.permutation(tmp2[:,l])
			
		imap = UMAP(n_neighbors = 5, min_dist = 0.1).fit_transform(tmp2.T)

		imap = (imap - imap.mean(0))/imap.std(0)

		data = pd.DataFrame(index = classe[good_sessions], data = imap)
		
		intra_dist_shuf[a] = []

		for i in np.unique(data.index):
			tmp2 = data.loc[i].values
			dist = np.sqrt(np.power(tmp2[:,[0]] - tmp2[:,0], 2) + np.power(tmp2[:,[1]] - tmp2[:,1], 2))
			intra_dist_shuf[a].append(dist[np.triu_indices_from(dist, 1)])
		intra_dist_shuf[a] = np.hstack(intra_dist_shuf[a])

		inter_dist_shuf[a] = []
		for i, j in combinations(range(4),2):
			tmp1 = data.loc[i].values
			tmp2 = data.loc[j].values	

			dist = np.sqrt(np.power(tmp1[:,[0]] - tmp2[:,0], 2) + np.power(tmp1[:,[1]] - tmp2[:,1], 2))
			inter_dist_shuf[a].append(dist.flatten())
		inter_dist_shuf[a] = np.hstack(inter_dist_shuf[a])

		ratios[a]['shuf'].append(intra_dist_shuf[a].mean()/inter_dist_shuf[a].mean())

clrs = ['red', 'blue']
figure(figsize = (10,3))
for i, a in enumerate(allreg.keys()):
	subplot(1,2,i+1)
	dist = ratios[a]['shuf']
	weights = np.ones_like(dist)/float(len(dist))
	hist(dist, 10, weights = weights, histtype = 'stepfilled')
	axvline(ratios[a]['true'], color = clrs[i], label = a)
	title(a)
savefig('../figures/figure_shuffling_rig_PSB.pdf', dpi = 200, bbox_inches = 'tight')

# figure()
# gs = GridSpec(2,2)
# for i,a in enumerate(allreg.keys()):
# # Intradistance
# 	subplot(gs[i,0])
# 	#dist = np.hstack([intra_dist_shuf[a] for i,a in enumerate(allreg.keys())])
# 	dist = intra_dist_shuf[a]
# 	weights = np.ones_like(dist)/float(len(dist))
# 	hist(dist, 10, weights = weights)
# 	clrs = ['red', 'blue']
# 	axvline(intra_dist[a].mean(), color = clrs[i], label = a)
		
# 	if i == 1: xlabel("Distance (a.u.)")
# 	if i == 0: title("Same envs")

# 	ylabel(a)

# 	subplot(gs[i,1])
# 	#dist = np.hstack([inter_dist_shuf[a] for i,a in enumerate(allreg.keys())])
# 	dist = inter_dist_shuf[a]
# 	weights = np.ones_like(dist)/float(len(dist))
# 	hist(dist, 10, weights = weights)
	
# 	axvline(inter_dist[a].mean(), color = clrs[i], label = a)
		
# 	if i == 0: title("Diff envs")
# 	if i == 1: xlabel("Distance (a.u.)")



# show()