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
from sklearn.manifold import Isomap
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


############################################################
# SELECTING DATA
############################################################
n_sessions_detected = np.sum(cellreg!=-1, 1)

# Detected in most sessions
tokeep = np.where(n_sessions_detected >= 10)[0]

# Good Cell reg scores 
tokeep = tokeep[scores[tokeep]>0.8]

# Selecting neurons with stable tuning curves
allst = {}
for i in tokeep:
	allst[i] = pd.Series(index = np.arange(cellreg.shape[1]), dtype = np.float32)
	for j in np.where(cellreg[i]!=-1)[0]:
		allst[i][j] = allinfo[list(allinfo.keys())[j]]['halfcorr'].loc[cellreg[i,j]]

allst = pd.concat(allst, 1).T
allst[allst<0.8] = np.nan
tokeep = allst[allst.notna().any(1)].index.values

# Selecting HD cells

#std[std<0.3] = np.nan
#tokeep = std[std.notna().any(1)].index.values


alltc = {}
for i in tokeep:
	alltc[i] = pd.DataFrame(columns = sessions)	
	for j in np.where(cellreg[i]!=-1)[0]:
		alltc[i][sessions[j]] = TC[list(TC.keys())[j]][cellreg[i,j]]

std = findSinglePeakHDCell(alltc, sessions)
std[std>0.7] = np.nan

tolook = [16,17]

tokeep = std.iloc[:,tolook].dropna().index.values

data5 = []
angles5 = []
idx5 = []
idx_sessions = cellreg[tokeep][:,tolook]

for k, s in enumerate(tolook):
	idx = idx_sessions[:,k]
	#idx = np.sort(cellreg[:,s][tokeep][cellreg[:,s][tokeep] > -1]) # neurons for session
	#idx = allinfo[s].loc[idx, 'peaks'].sort_values().index
	# tc = TC[s][idx].copy()
	# tc = centerTuningCurves(tc)
	#nidx = idx[np.argsort(std.iloc[:,s].dropna().values)]

	nt = DFFS[s].shape[0]

	data = np.zeros((nt,len(idx)))

	angles = downsampleAngleFromC(DFFS[s].index.values, positions[s]['ry'])
	angles = angles.values

	for j, n in enumerate(idx):
		tmp = DFFS[s][n].values
		# if cellreg[n,i] != -1:
			# tmp = DFFS[i][cellreg[n,i]].values[0:nt]
		#tmp = np.sqrt(tmp)
		# tmp = tmp - tmp.mean()
		# tmp = tmp / tmp.std()
		#tmp = scipy.ndimage.gaussian_filter1d(tmp, 1)
		#data[:,j] = scipy.signal.decimate(tmp, decim)
		data[:,j] = tmp

	# data[data<data.max(0)/3] = 0.0
	# data[data>0.0] = 1.0

	# filtering
	data2 = scipy.ndimage.gaussian_filter1d(data, sigma = 30, axis = 0)

	# downsamplign
	idx3 = np.arange(0, data2.shape[0], 10)
	data3 = data2[idx3]
	angles3 = angles[idx3]

	# pick most active bins
	idx4 = data3.sum(1) > np.percentile(data3.sum(1), 60)
	#idx2 = np.sum(data==0,1) < data.shape[1]*0.8
	#idx2 = data.sum(1) > 0.0
	data4 = data3[idx4]
	angles4 = angles3[idx4]

	data4 = data4 - data4.mean(0)
	data4 = data4 / data4.std(0)

	data5.append(data4)
	angles5.append(angles4)
	idx5.append(np.ones(len(data4))*k)

data5 = np.vstack(data5)
angles5 = np.hstack(angles5)
idx5 = np.hstack(idx5)

H = angles5/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)


imap = Isomap(n_neighbors = 100, n_components = 3).fit_transform(data5)

imap = UMAP(n_neighbors = 100, min_dist = 1, low_memory=True).fit_transform(data4)

#imap = PCA(n_components = 2).fit_transform(data)

figure()
scatter(imap[:,0], imap[:,1], c= RGB, marker = '.', alpha = 0.5, linewidth = 0, s = 100)
show()

# sys.exit()

from mpl_toolkits import mplot3d

fig = figure()
ax = axes(projection='3d')
markers = ['o', '^']
colors = ['blue', 'green']
for i in range(2):	
#	ax.scatter(imap[idx5==i,0], imap[idx5==i,1], imap[idx5==i,2], color = RGB[idx5==i], marker = markers[i])
	ax.scatter(imap[idx5==i,0], imap[idx5==i,1], imap[idx5==i,2], color = colors[i])
show()



figure()
subplot(111)
for i, n in enumerate(idx):
	#plot(allinfo[s].loc[n,'peaks'] + data[:,i]*0.5)
	plot(allinfo[s].loc[n,'peaks'] + data2[:,i]*2)
plot(angles)

show()

figure()
subplot(111)
for i, n in enumerate(idx):
	#plot(allinfo[s].loc[n,'peaks'] + data[:,i]*0.5)
	plot(allinfo[s].loc[n,'peaks'] + data3[:,i]*2)
plot(angles3)

show()

# figure()
# for i, n in enumerate(tokeep):
# 	subplot(5,10,i+1)#,projection='polar')
# 	plot(alltc[n].iloc[:,s])
# 	title(std.iloc[:,s].loc[n])
# 	xticks([])
# 	yticks([])