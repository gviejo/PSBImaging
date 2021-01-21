import numpy as np
import sys,os
import scipy.io
import neuroseries as nts
import pandas as pd
import scipy.signal
from numba import jit
import h5py
'''
Wrappers functions for miniscope data
'''

def loadCalciumData(path, fs =30, dims = (304, 304)):
	"""

	"""    
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()    
	new_path = os.path.join(path, 'Analysis/')
	if os.path.exists(new_path):		
		files        = os.listdir(new_path)
		# TODO MAKE CONDITIONS
		C = pd.read_hdf(os.path.join(new_path, 'C.h5'))
		A = pd.read_hdf(os.path.join(new_path, 'A.h5'))
		position = pd.read_hdf(os.path.join(new_path, 'position.h5'))
		position.columns = ['ry', 'rx', 'rz', 'x', 'y', 'z']
		A = A.values
		A = A.T.reshape(A.shape[1], dims[0], dims[1])		

		return A, nts.TsdFrame(C), nts.TsdFrame(position)

	files = os.listdir(path)

	# laoding calcium trnasients
	C_file = [f for f in files if 'C' in f][0]
	if len(C_file):
		C = pd.read_csv(os.path.join(path, C_file), header = None)
		C = C.T
	else:
		print("No calcium transient csv file in ", path)
		sys.exit()

	# aligning C with time stamps from analogin.dat
	analogin_file = [f for f in files if 'analogin' in f]
	if len(analogin_file):
		f = open(os.path.join(path, analogin_file[0]), 'rb')
		startoffile = f.seek(0, 0)
		endoffile = f.seek(0, 2)
		bytes_size = 2        
		n_samples = int((endoffile-startoffile)/2/bytes_size)
		f.close()
		n_channels = 2
		with open(os.path.join(path, analogin_file[0]), 'rb') as f:
			analogin = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
		analogin = analogin.astype(np.int32)
		timestep = np.arange(0, len(analogin))/20000

		peaks,_ = scipy.signal.find_peaks(np.diff(analogin[:,0]), height=30000)		
		peaks+=1
		ttl_tracking = pd.Series(index = timestep[peaks], data = analogin[peaks,0])

		peaks,_ = scipy.signal.find_peaks(np.abs(np.diff(analogin[:,1])), height=30000, distance = 500)
		ttl_miniscope = pd.Series(index = timestep[peaks], data = analogin[peaks,1])
	else:
		print("No analogin file in ", path)
		sys.exit()

	C = C.iloc[0:np.minimum(len(ttl_miniscope),len(C))]
	C.index = nts.Ts(t = ttl_miniscope.index.values[0:np.minimum(len(ttl_miniscope),len(C))], time_units = 's').index

	C = nts.TsdFrame(C)

	# aligning position
	csv_file = [f for f in files if '_0.csv' in f][0]
	if len(csv_file):
		position = pd.read_csv(os.path.join(path, csv_file), header = [4,5], index_col = 1)
		if 1 in position.columns:
			position = position.drop(labels = 1, axis = 1)
		position = position[~position.index.duplicated(keep='first')]
		position.columns = ['ry', 'rx', 'rz', 'x', 'y', 'z']
		position[['ry', 'rx', 'rz']] *= (np.pi/180)
		position[['ry', 'rx', 'rz']] += 2*np.pi
		position[['ry', 'rx', 'rz']] %= 2*np.pi
	else:
		print("No position file in ", path)
		sys.exit()

	position.index = nts.Ts(t = ttl_tracking.index[0:len(position)].values, time_units = 's').index
	position = nts.TsdFrame(position)	
	# bad


	# loading spatial footprints
	A_file = [f for f in files if '_A.csv' in f][0]
	if len(A_file):		
		A = pd.read_csv(os.path.join(path, A_file), header = None)
		
	else:
		print("No calcium transient csv file in ", path)
		sys.exit()

	# Cutting C and position to be the same duration
	ep = nts.IntervalSet(start = np.maximum(C.index.values[0], position.index.values[0]),
						end = np.minimum(C.index.values[-1], position.index.values[-1]))

	C = C.restrict(ep)
	position = position.restrict(ep)

	# saving
	# Creating /Analysis/ Folder here if not already present
	# if not os.path.exists(new_path): os.makedirs(new_path)

	# C.as_dataframe().to_hdf(os.path.join(new_path, 'C.h5'), 'C', format='table')
	# position.as_dataframe().to_hdf(os.path.join(new_path, 'position.h5'), 'p', format='table')
	# A.to_hdf(os.path.join(new_path, 'A.h5'), 'A', format='table')

	A = A.values
	A = A.T.reshape(A.shape[1], dims[0], dims[1])

	position.columns = ['ry', 'rx', 'rz', 'x', 'y', 'z']

	return A, C, position

def loadCellReg(path, folder_name = 'CellReg', file_name = 'cellRegistered'):
	folder_path = os.path.join(path, folder_name)
	assert os.path.exists(folder_path)
	files = os.listdir(folder_path)	
	cellreg_file = np.sort([f for f in files if file_name in f])[-1]
	assert len(cellreg_file)
	
	arrays = {}
	f = h5py.File(os.path.join(folder_path, cellreg_file))
	for k, v in f.items():
	    arrays[k] = v
	cellreg = np.copy(np.array(arrays['cell_registered_struct']['cell_to_index_map']))
	f.close()
	cellreg = cellreg.T - 1 
	return cellreg.astype(np.int)