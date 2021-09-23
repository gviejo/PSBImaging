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
from matplotlib.gridspec import GridSpec


data_directory = '/mnt/DataRAID/MINISCOPE'
#data_directory = '/media/guillaume/Elements'

############################################################

fbasename = 'A6509'
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
paths = [os.path.join(data_directory, fbasename[0:3] + '00', fbasename, fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
sessions = [fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '') for i in info.index]
info['paths'] = paths
info['sessions'] = sessions
info = info.set_index('sessions')

path = os.path.join(data_directory, fbasename[0:3] + '00', fbasename, 'CellRegRefs')

files = [f for f in os.listdir(path) if '.mat' in f]

logs = [f for f in os.listdir(path) if '.txt' in f]


cellregref = {}
scoresref = {}
mean_scores = []
num_cells = []
ratio = []
total_reg = []

for i in range(len(info)):
	arrays = {}
	matfile = h5py.File(os.path.join(path, 'cellRegistered_'+str(i+1)+'.mat'), 'r')
	for k, v in matfile.items():
	    arrays[k] = v
	cellreg = np.copy(np.array(arrays['cell_registered_struct']['cell_to_index_map']))
	scores = np.copy(np.array(arrays['cell_registered_struct']['cell_scores']))
	matfile.close()
	cellreg = cellreg.T - 1 
	cellreg = cellreg.astype(np.int)
	scores = scores.flatten()

	cellregref[i] = cellreg
	scoresref[i] = scores
	mean_scores.append(np.mean(scores[scores<1]))
	num_cells.append(cellreg.shape[0])
	ratio.append(np.sum(cellreg>-1)/np.sum(cellreg==-1))

	total = []

	for j in range(cellreg.shape[1]):
		tmp = cellreg[cellreg[:,j]>-1,:]
		total.append(np.sum(tmp>-1,1)/cellreg.shape[1])

	total_reg.append(np.hstack(total))


mean_total_reg = [np.mean(tmp) for tmp in total_reg]

plot(mean_total_reg, 'o')