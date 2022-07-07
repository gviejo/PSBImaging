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


#####################################################
#####################################################

envs = info.loc[sessions].groupby('Rig').groups

order = ['Circular', 'Square', '8-arm maze', 'Open field']

sessions = list(sessions)

ref = sessions[np.sum(cellreg[tokeep] == -1, 0).argmin()]# reference sessions

norder = allpk[ref].dropna().sort_values().index.values

figure(figsize = (16, 1))
gs = GridSpec(1, len(sessions), wspace = 0.2, top = 0.95, bottom = 0.05, left = 0.03, right = 0.96)

# plot matrix of tc
for j, s in enumerate(sessions):		
	tc = np.array([alltc[n][s].values for k, n in enumerate(norder) if ~np.isnan(alltc[n][s].values[0])])
	#tc = (tc.T/tc.max(1)).T
	tc2 = gaussian_filter(tc, 2)	
	if len(tc2):	
		ax = subplot(gs[0,j], aspect = 'equal')
		#noaxis(ax)
		imshow(tc2, cmap = 'jet', aspect = 'auto')
		yticks([])
		xticks([])
		
		gca().text(0.05, 0.05, len(tc2)-1 , transform = gca().transAxes)
savefig('../figures/figure_psb_'+fbasename+'_13days.pdf', format = 'pdf', dpi = 200, bbox_inches = 'tight')
