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
from matplotlib import colors

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

if fbasename == 'A0634':
	dims = (166, 136)
elif fbasename == 'A6509':
	dims = (202,192)


############################################################
# LOADING DATA
############################################################
SF, TC, PF, allinfo = loadDatas(paths, dims)


cellreg, scores = loadCellReg(os.path.join(data_directory, fbasename[0:3] + '00', fbasename))


# sys.exit()
# # unifying cellreg and datas
# cellreg = cellreg[:,list(TC.keys())]
# scores = scores[list(TC.keys())]
# sessions = np.array(sessions)[list(TC.keys())]



############################################################

n_sessions_detected = np.sum(cellreg!=-1, 1)

# selecting neurons detected in more than 3 sessions
tokeep = np.where(n_sessions_detected == 8)[0]

allst = {}
for i in tokeep:
	allst[i] = pd.Series(index = np.arange(cellreg.shape[1]), dtype = np.float32)
	for j in np.where(cellreg[i]!=-1)[0]:
		allst[i][j] = allinfo[list(allinfo.keys())[j]]['halfcorr'].loc[cellreg[i,j]]

allst = pd.concat(allst, 1).T



# Selecting neurons with stable tuning curves
allst[allst<0.3] = np.nan

tokeep = allst[allst.notna().any(1)].index.values




alltc = {}
allpf = {}
for i in tokeep:
	alltc[i] = pd.DataFrame(columns = sessions)
	allpf[i] = np.zeros((len(sessions), PF[0].shape[1], PF[0].shape[2],))
	for j in np.where(cellreg[i]!=-1)[0]:

		alltc[i][sessions[j]] = TC[list(TC.keys())[j]][cellreg[i,j]]
		allpf[i][j] = PF[list(TC.keys())[j]][cellreg[i,j]]



#####################################################################################
# PLOT ALL TUNING CURVES
#####################################################################################
index = np.array(list(alltc.keys()))[0:20]

rigs = ['Circular', 'Square']#, '8-arm maze', 'Open field']

figure()
for i, n in enumerate(index):	
	ax = subplot(int(np.ceil(np.sqrt(len(index))))-1,int(np.ceil(np.sqrt(len(index)))),i+1)
	gs = GridSpecFromSubplotSpec(1,len(rigs),ax)
	tmp = alltc[n].dropna(1, 'all')
	grp = info.loc[tmp.columns].groupby('Rig').groups
	for j, m in enumerate(rigs):
		if m in grp.keys():
			subplot(gs[0,j], projection = 'polar')
			plot(alltc[n][grp[m]], color = cm.rainbow(np.linspace(0,1,len(rigs)))[j])
			xticks([])
			yticks([])
			title(n)

figure()

subplot(121)
A = SF[0]
af = np.zeros((A.shape[0], dims[0], dims[1]))
for i in range(A.shape[0]):
	af[i] = A[i]

peaks = allinfo[0]['peaks']

H = peaks.values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)

colorA = np.zeros((dims[0], dims[1], 3))
colorA *= np.nan

for i in range(len(af)):
	colorA[af[i] > 4] = RGB[i]

imshow(colorA)
xticks([])
yticks([])

ax = subplot(122)
gs = GridSpecFromSubplotSpec(3,2, ax)
toplot = [24, 15, 8]
s1oplot = [0, 0, 0]
s2oplot = [4, 5, 4]
for i in range(3):
	subplot(gs[i,0], projection = 'polar')
	plot(alltc[toplot[i]].iloc[:,s1oplot[i]])
	xticks([])
	yticks([])
	subplot(gs[i,1], projection = 'polar')
	plot(alltc[toplot[i]].iloc[:,s2oplot[i]])
	xticks([])
	yticks([])


savefig('../figures/figure_adrian_1.pdf', format = 'pdf', dpi = 200, bbox_inches = 'tight')

figure()

pair = [15, 53]

subplot(211)
A = SF[0]
af = np.zeros((A.shape[0], dims[0], dims[1]))
for i in range(A.shape[0]):
	af[i] = A[i]

colorA = np.zeros((dims[0], dims[1], 4))
colorA *= np.nan

for i in range(len(af)):
	colorA[af[i]> 4] = colors.to_rgba('lightsteelblue')	

colorA[af[pair[0]] > 3] = colors.to_rgba('orange')
colorA[af[pair[1]] > 3] = colors.to_rgba('green')	

imshow(colorA)
xticks([])
yticks([])

ax = subplot(212)
gs = GridSpecFromSubplotSpec(2,4, ax)
clrs = ['orange', 'green']
for i in range(4):
	for j, n in enumerate(pair):
		subplot(gs[j,i], projection = 'polar')
		print(n, i)
		plot(alltc[n].iloc[:,i], color = clrs[j])
		xticks([])
		yticks([])

savefig('../figures/figure_adrian_2.pdf', format = 'pdf', dpi = 200, bbox_inches = 'tight')

figure()
gs = GridSpec(10, 4)
for i, n in enumerate(tokeep[30:40]):
	for j in range(4):
		subplot(gs[i,j], projection = 'polar')
		print(n, j)
		plot(alltc[n].iloc[:,j])
		title(n)
		xticks([])
		yticks([])


