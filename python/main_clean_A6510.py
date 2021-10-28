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


data_directory = '/mnt/DataRAID/MINISCOPE'

############################################################
# ANIMAL INFO
############################################################
fbasename = 'A0643'
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
paths = [os.path.join(data_directory, fbasename[0:3] + '00', fbasename, fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
sessions = [fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '') for i in info.index]
info['paths'] = paths
info['sessions'] = sessions
info = info.set_index('sessions')

if fbasename == 'A6509':
	dims = (202,192)
elif fbasename == 'A0643':
	dims = (186,186)
elif fbasename == 'A6510':
	dims = (192,251)
elif fbasename == 'A6512':
	dims = (182,285)

############################################################
# LOADING DATA
############################################################
SF, TC, PF, allinfo, positions, DFFS, Cs = loadDatas(paths, dims)
#sys.exit()

print('\n')

bad_sessions = []
for i in SF.keys():
	if np.isnan(np.sum(SF[i])):
		print(sessions[i])
		bad_sessions.append(sessions[i])
		good_neurons = np.where(~np.isnan(SF[i]).any((1,2)))[0]
		A = SF[i][good_neurons]
		C = Cs[i][good_neurons].values

		A = A.reshape(len(good_neurons), np.prod(dims))
		A = A.T
		C = C.T
		p = os.path.join(info.iloc[i]['paths'], sessions[i])
		np.savetxt(p+'_A.csv', A, delimiter = ',')
		np.savetxt(p+'_C.csv', C, delimiter = ',')

		np.savetxt(p+'_good_neurons.txt', np.vstack(good_neurons),fmt='%i')


np.savetxt('/mnt/DataRAID/MINISCOPE/'+fbasename[0:3]+'00/'+fbasename+'/bad_sessions.txt', bad_sessions, fmt='%s')
