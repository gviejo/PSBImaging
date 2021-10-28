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

	
	# # Selecting HD cells
	# alltc = {}
	# for i in tokeep: # neuron index
	# 	alltc[i] = pd.DataFrame(columns = sessions)
	# 	for j in np.where(cellreg[i]!=-1)[0]:
	# 		alltc[i][sessions[j]] = TC[list(TC.keys())[j]][cellreg[i,j]]

	# std = findSinglePeakHDCell(alltc, sessions)
	#std[std>0.7] = np.nan
	#tokeep = std.dropna(0).index.values


####################################################
# CLUSTERING BASED ON CELL REG	
####################################################
tmp = cellreg.copy()
tmp[tmp > -1] = 1
imap = UMAP(n_neighbors = 5, min_dist = 0.1, low_memory=True).fit_transform(tmp.T)




#####################################################
# COMMON CELLS
#####################################################
tmp = cellreg.copy()
tmp[tmp > -1] = 1
corr = np.zeros((tmp.shape[1],tmp.shape[1]))

for i,j in combinations(np.arange(tmp.shape[1]), 2):
	corr[i,j] = np.sum(np.prod(tmp[:,[i,j]], 1)==1) / tmp.shape[0]
	corr[j,i] = corr[i,j]


figure()
subplot(221)
scatter(imap[:,0], imap[:,1], c = np.arange(0, len(imap)))

subplot(222)
imshow(corr)

subplot(223)
for i in range(tmp.shape[1]):
	plot(np.arange(i+1, tmp.shape[1]), corr[i,i+1:])

show()