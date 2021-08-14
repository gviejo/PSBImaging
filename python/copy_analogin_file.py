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
import cv2
from matplotlib.gridspec import GridSpec
from itertools import product, combinations
from scipy.stats import norm

from shutil import copy2



data_directory = '/mnt/DataRAID/MINISCOPE'
# info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_A0642.txt', comment = '#', header = None)

fbasename 		= 'A0634'
info 			= pd.read_csv('/home/guillaume/PSBImaging/python/datasets_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
paths 			= [os.path.join(data_directory, fbasename[0:3]+'00', fbasename, 'minian', fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
info['paths'] 	= paths





for i in range(len(info)):
	path 			= info.loc[i,'paths']	
	if os.path.exists(path):
		name 			= os.path.basename(path)
		path2 = os.path.join('/mnt/Hypernova/Guillaume/A0600', fbasename, name)
		try:
			copy2(path2+'/'+name+'_0.csv', path)
			copy2(path2+'/'+name+'_0_analogin.dat', path)
		except:
			print(name)

		#print(name)

