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


#data_directory = '/mnt/DataRAID/MINISCOPE'
data_directory = '/media/guillaume/Elements'

############################################################
# ANIMAL INFO
############################################################
fbasename = 'A0634'
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
paths = [os.path.join(data_directory, fbasename[0:3] + '00', fbasename, fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
sessions = [fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '') for i in info.index]
info['paths'] = paths
info['sessions'] = sessions
info = info.set_index('sessions')

if fbasename == 'A0634':
	dims = (166, 136)


############################################################
# LOADING DATA
############################################################
SF, TC, PF, allinfo = loadDatas(paths, dims)


cellreg, scores = loadCellReg(os.path.join(data_directory, fbasename[0:3] + '00', fbasename))

# unifying cellreg and datas
cellreg = cellreg[:,list(TC.keys())]
scores = scores[list(TC.keys())]

sessions = np.array(sessions)[list(TC.keys())]



############################################################

n_sessions_detected = np.sum(cellreg!=-1, 1)

tokeep = np.where(n_sessions_detected > 3)[0]

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

rigs = ['Circular', 'Square', '8-arm maze', 'Open field']

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





show()









