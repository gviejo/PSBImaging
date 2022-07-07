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


for a in ['A0642']:
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
	imap = UMAP(n_neighbors = 5, min_dist = 0.1).fit_transform(tmp[:,good_sessions].T)
	#imap = TSNE(n_components = 2, perplexity = 1).fit_transform(tmp.T)
	#imap = Isomap(n_components = 2, n_neighbors = 20).fit_transform(tmp.T)
	classe = np.array([list(order).index(e) for e in infos[a]['Rig'].values])
	data = pd.DataFrame(index = classe[good_sessions], data = imap)
	clus[a] = data



##################################################################################
# FIGURES
##################################################################################


matplotlib.rcParams.update({'font.size': 14})

figure(figsize = (18, 6))
gs = GridSpec(1,3,width_ratios = [3,1,2])



################
# Example
subplot(gs[0,0])
groups = infos[a].groupby('Rig').groups
examples = [groups[e][i] for e,i in zip(groups.keys(),[2,2,0,0])]
idxs = np.array([np.where(infos[a].index == s)[0][0] for s in examples])
# find 2 neurons only in each sessions
tmp = allreg[a][:,idxs].copy()

unique_session = {}
for i in range(4):
	idx = np.intersect1d(np.where(tmp[:,i] != -1)[0],np.where(np.all(np.delete(tmp, i, 1) == -1, 1))[0])[0:3]
	unique_session[idxs[i]] = tmp[idx,i]


subgs = GridSpecFromSubplotSpec(3,2, gs[0,0], height_ratios = [0.2, 0.4, 0.2])


cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

tolook = np.array([0,1])

for i, s in enumerate(idxs[tolook]):
	#subplot(subgs[i//2,i%2])
	subplot(subgs[0,i%2])
	plot(positions[s]['x'], positions[s]['z'], alpha = 0.7)
	noaxis(gca())
	gca().set_aspect('equal')

	subplot(subgs[1,i%2])
	title(infos[a].loc[examples[i]].Rig)
	xticks([])
	yticks([])
	A = SF[s]

	Aall = np.zeros(A.shape[1:])
	for j in range(len(A)):
		tmp = A[j]
		tmp = tmp / tmp.max()
		idx = tmp>0.6
		Aall[idx] = Aall[idx] + tmp[idx]
		
	jet = get_cmap('jet')
	cNorm = colors.Normalize(vmin = 0, vmax = Aall.max())
	scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

	colorVal = scalarMap.to_rgba(Aall)

	A2 = SF[s][unique_session[s]]
	for j in range(len(A2)):
		tmp = A2[j]
		tmp = tmp/tmp.max()
		idx = tmp > 0.3
		colorVal[idx] = colors.to_rgba(cycle_colors[i])

	# adding missing neurones
	s2 = idxs[tolook][np.arange(2)[::-1][i]]
	A2 = SF[s2][unique_session[s2]]
	for j in range(len(A2)):
		tmp = np.zeros_like(A2[j])
		pos = np.array(np.unravel_index(np.argmax(A2[j]), tmp.shape))
		if i == 0 and j == 1: pos[0] -= 5
		tmp[tuple(pos)] = 1
		tmp = gaussian_filter(tmp, 5)
		tmp = tmp / tmp.max()
		idx = np.logical_and(tmp > 0.4, tmp<0.6)
		colorVal[idx] = colors.to_rgba(cycle_colors[np.arange(2)[::-1][i]])
		#colorVal[idx] = colors.to_rgba('red')



	imshow(colorVal)

	xlim(50,209)
	ylim(70,192)

	# TUNING CURVES
	gstc = GridSpecFromSubplotSpec(1,3, subgs[2,i%2])
	for j,n in enumerate(unique_session[s]):
		subplot(gstc[0,j], projection='polar')
		plot(TC[s][n])
		xticks([])



################
# Binarized matrix
subplot(gs[0,1])

for a in allreg.keys():
	cellreg = allreg[a]
	tmp = cellreg.copy()
	tmp[tmp > -1] = 1
	p = np.sum(tmp==-1, 0)/len(tmp)
	good_sessions = np.where(p < 0.5)[0]
	tmp = tmp[:,good_sessions]

imshow(tmp*-1, cmap = 'binary', aspect = 'equal')
tmp = infos[a]['Rig'].iloc[good_sessions]
for s in np.unique(infos[a]['Rig']):
	axvline(np.where(tmp == s)[0][0]-0.5 , color = 'red')
	axvline(np.where(tmp == s)[0][-1]+0.5, color = 'red')

title("Cell registration matrix ")
xlabel("Sessions")
ylabel("Neurons")

############### 
# DIFFERENT ENVS

ordercolorenv = dict(zip(groups.keys(), cycle_colors[0:4]))


for j, a in enumerate(clus.keys()):
	ax = subplot(gs[0,2])
	simpleaxis(ax)
	#color = iter(cm.rainbow(np.linspace(0, 1, len(order))))	
	classe = clus[a].index.values
	imap = clus[a].values		
	for k in range(4):
		scatter(imap[classe==k,0], imap[classe==k,1], color = ordercolorenv[order[k]], label = order[k])
	legend()		
	#title(a)
	xlabel("UMAP component 1")
	ylabel("UMAP component 2")


savefig('../figures/figure_RIG_decoding_'+a+'.pdf', dpi = 200, bbox_inches = 'tight')

#show()


