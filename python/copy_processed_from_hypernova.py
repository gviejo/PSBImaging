import numpy as np
import sys, os
from shutil import copy2, copytree
import pandas as pd

data_directory = '/mnt/DataRAID/MINISCOPE'
#data_directory = '/media/guillaume/Elements'

############################################################

fbasename = 'A0642'
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_same_env_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
paths = [os.path.join(data_directory, fbasename[0:3] + '00', fbasename, fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
sessions = [fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '') for i in info.index]


path = '/mnt/Hypernova/Guillaume/A0600/A0642'
target = '/mnt/DataRAID/MINISCOPE/A0600/A0642'

for s in sessions:
	#print(l)
	p = os.path.join(path, s)
	if os.path.isdir(p):
		pt = os.path.join(target, s)
		if not os.path.exists(pt):
			os.mkdir(pt)
		lf1 = os.listdir(p)
		lf2 = os.listdir(pt)
		tocopy = list(set(lf1) - set(lf2))		
		for t in tocopy:							
			po = os.path.join(p, t)
			print(po)
			if t != 'Miniscope' and 'tak' not in t:
				if os.path.isdir(po):
					copytree(po, os.path.join(pt, t))
				else:
					copy2(po, pt)