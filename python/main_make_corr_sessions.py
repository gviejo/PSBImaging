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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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

datas = {}

stability = {}

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

	stability[a] = allst.copy()

	allst[allst<0.1] = np.nan
	tokeep = allst[allst.notna().any(1)].index.values

	print(len(tokeep))

	decim = 3
	nt = np.min([DFFS[i].shape[0] for i in Cs.keys()]) 
	ntm = int(np.ceil(nt/decim))
	# MAKING THE DATA
	data = np.zeros((ntm,len(tokeep),len(sessions)))	
	for i in Cs.keys():
		for j, n in enumerate(tokeep):
			if cellreg[n,i] != -1:
				tmp = Cs[i][cellreg[n,i]].values[0:nt]
				tmp = np.sqrt(tmp)
				# tmp = tmp - tmp.mean()
				# tmp = tmp / tmp.std()
				tmp = scipy.ndimage.gaussian_filter1d(tmp, 80)			
				data[:,j,i] = scipy.signal.decimate(tmp, decim)
				
	datas[a] = data

#######################################################################
# PAIRWISE CORRELATION
#######################################################################
ccs = {}

for a in datas.keys():
	cc = [] # inter neurons
	for i in range(datas[a].shape[-1]):
		tmp1 = np.corrcoef(datas[a][:,:,i].T)
		tmp2 = tmp1[np.triu_indices_from(tmp1,k=1)]
		cc.append(tmp2) 
	cc = np.array(cc).T

	cc2 = np.zeros((cc.shape[1],cc.shape[1])) #inter sessions

	for i,j in combinations(range(cc.shape[1]),2):
		idx = ~np.isnan(cc[:,(i,j)]).any(1)
		# print(np.sum(idx)/len(idx))
		cc2[i,j] = np.corrcoef(cc[:,(i,j)][idx].T)[0,1]
		cc2[j,i] = cc2[i,j]

	ccs[a] = cc2

#######################################################################
# TSNE 
#######################################################################


#######################################################################
# FIGURES
#######################################################################
figure(figsize = (17, 10))
gs = GridSpec(2,2)

grps = struct.groupby('struct').groups

for i, g in enumerate(grps.keys()):
	for j, a in enumerate(grps[g]):
		gs2 = GridSpecFromSubplotSpec(1,2,gs[i,j])
		subplot(gs2[0,0])
		imshow(ccs[a])
		for s in np.unique(infos[a]['Rig']):
			axvline(np.where(infos[a]['Rig'] == s)[0][0]-0.5 , color = 'red')
			axvline(np.where(infos[a]['Rig'] == s)[0][-1]+0.5, color = 'red')
			axhline(np.where(infos[a]['Rig'] == s)[0][0]-0.5 , color = 'red')
			axhline(np.where(infos[a]['Rig'] == s)[0][-1]+0.5, color = 'red')
		title(a)
		subplot(gs2[0,1])
		title(a)
		envs = infos[a].groupby('Rig').groups		
		sessions = infos[a].index.values
		xcorr = pd.DataFrame(index = sessions, columns = sessions, data = ccs[a])
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

savefig('../figures/figure_envs_correlation_all.pdf', dpi = 200, bbox_inches = 'tight')


figure(figsize = (14, 6))
subplot(121)
for i, a in enumerate(stability.keys()):	
	tmp = stability[a].values.flatten()	
	tmp = tmp[~np.isnan(tmp)]
	hist(tmp, weights = np.ones_like(tmp)/len(tmp), alpha = 0.4, label = a)
legend()
xlabel("Stability")
subplot(122)
order = ['Circular', 'Square', '8-arm maze', 'Open field']
symbols = ['*-', '^-', 'o-', 's-']
colors = ['red', 'blue', 'orange', 'magenta']
for i, a in enumerate(stability.keys()):
	grps = infos[a].groupby('Rig').groups
	tmp = np.nanmean(stability[a], 0)
	plot(tmp)
	# count = 0
	# for j, e in enumerate(order):
	# 	plot(np.arange(count, count+len(grps[e])), tmp[count:count+len(grps[e])], symbols[j], color= colors[j])		
	# 	count += len(grps[e])

#legend()
xlabel("Session")
ylabel("Stability")

savefig('../figures/figure_stability_all.pdf', dpi = 200, bbox_inches = 'tight')

