import numpy as np
import sys,os
import scipy.io
import neuroseries as nts
import pandas as pd
import scipy.signal
from numba import jit
import h5py
from functions import *


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
		files = os.listdir(new_path)
		if len(files) == 4: # BAD
			
			C = pd.read_hdf(os.path.join(new_path, 'C.h5'))
			DFF = pd.read_hdf(os.path.join(new_path, 'DFF.h5'))
			position = pd.read_hdf(os.path.join(new_path, 'position.h5'))
			A = np.load(os.path.join(new_path, 'A.npy'))

			return A, C, DFF, position


	files = os.listdir(path)

	# loading calcium transients
	C_file = [f for f in files if 'C.csv' in f][0]
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

		# determining automatically which channel is tracking
		peaks,_ = scipy.signal.find_peaks(np.diff(analogin[:,0]), height=30000)
		peaks+=1
		ttl0 = pd.Series(index = timestep[peaks], data = analogin[peaks,0])
		peaks,_ = scipy.signal.find_peaks(np.diff(analogin[:,1]), height=30000)
		peaks+=1
		ttl1 = pd.Series(index = timestep[peaks], data = analogin[peaks,0])
		if np.mean(np.diff(ttl0.index)) < np.mean(np.diff(ttl1.index)): # channel 0 is tracking
			ttl_tracking = ttl0
			peaks,_ = scipy.signal.find_peaks(np.abs(np.diff(analogin[:,1])), height=30000, distance = 500)
			ttl_miniscope = pd.Series(index = timestep[peaks], data = analogin[peaks,1])
		else:
			ttl_tracking = ttl1
			peaks,_ = scipy.signal.find_peaks(np.abs(np.diff(analogin[:,0])), height=30000, distance = 500)
			ttl_miniscope = pd.Series(index = timestep[peaks], data = analogin[peaks,0])

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
		names = []
		for n in position.columns:
			if n[0] == 'Rotation':
				names.append('r'+n[1].lower())
			elif n[0] == 'Position':
				names.append(n[1].lower())
			else:
				print('Unknow csv file for position; Exiting')
				sys.exit()

		position.columns = names
		position = position[['ry', 'rx', 'rz', 'x', 'y', 'z']]
		position[['ry', 'rx', 'rz']] *= (np.pi/180)
		position[['ry', 'rx', 'rz']] += 2*np.pi
		position[['ry', 'rx', 'rz']] %= 2*np.pi
	else:
		print("No position file in ", path)
		sys.exit()

	position = position.fillna(0)

	# REALING POSITION
	# NEED TO CHECK FOR MISSING TTLS
	# ASSUMING TTL_tracking should be between 80 and 140 Hz according to motive
	if np.max(np.diff(ttl_tracking.index.values)) < 1/80 and np.min(np.diff(ttl_tracking.index.values)) > 1/140:
		position.index = nts.Ts(t = ttl_tracking.index[0:len(position)].values, time_units = 's').index
		position = nts.TsdFrame(position)	
		# bad
	else: # PRoblem with the ttls
		print("Problem with the tracking in ", path)
		tmp = np.diff(ttl_tracking.index)
		# FILLING MISSING TTLS ONLY IF ONE MISSING TTLS		
		if np.max(tmp) < (1/100) * 3:
			tmp2 = np.array([ttl_tracking.index[i]+tmp[i]/2 for i in range(len(tmp)) if tmp[i] > 1/80])
			ttl_tracking = ttl_tracking.append(pd.Series(index=tmp2,data=np.nan)).sort_index()
			position.index = nts.Ts(t = ttl_tracking.index[0:len(position)].values, time_units = 's').index
			position = nts.TsdFrame(position)	

		else:
			print("Big problem with the tracking in ", path)
			sys.exit()




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

	A = A.values
	A = A.T.reshape(A.shape[1], dims[0], dims[1])

	position.columns = ['ry', 'rx', 'rz', 'x', 'y', 'z']

	DFF 			= C.diff()
	DFF 			= DFF.fillna(0).as_dataframe()
	DFF[DFF<0]		= 0	

	# saving
	# Creating /Analysis/ Folder here if not already present
	if not os.path.exists(new_path): os.makedirs(new_path)
	C.as_dataframe().to_hdf(os.path.join(new_path, 'C.h5'), 'C', format='table')
	DFF.to_hdf(os.path.join(new_path, 'DFF.h5'), 'DFF', format = 'table')
	position.as_dataframe().to_hdf(os.path.join(new_path, 'position.h5'), 'p', format='table')	
	np.save(os.path.join(new_path, 'A.npy'), A)

	return A, C, DFF, position

def loadCellReg(path, folder_name = 'CellReg', file_name = 'cellRegistered'):
	folder_path = os.path.join(path, folder_name)
	assert os.path.exists(folder_path)
	files = os.listdir(folder_path)	
	cellreg_file = np.sort([f for f in files if file_name in f])[-1]
	assert len(cellreg_file)
	
	arrays = {}
	f = h5py.File(os.path.join(folder_path, cellreg_file), 'r')
	for k, v in f.items():
	    arrays[k] = v
	cellreg = np.copy(np.array(arrays['cell_registered_struct']['cell_to_index_map']))
	scores = np.copy(np.array(arrays['cell_registered_struct']['cell_scores']))
	f.close()
	cellreg = cellreg.T - 1 
	scores = scores.flatten()
	return cellreg.astype(np.int), scores

def loadDatas(paths, dims):
	SF = {}
	TC = {}
	PF = {}
	allinfo = {}
	positions = {}
	DFFS = {}
	Cs = {}

	for i, s in enumerate(paths):
		print(s)
		name 			= s.split('/')[-1]
		
		A, C, DFF, position 	= loadCalciumData(s, dims = dims)
		tuningcurve		= computeCalciumTuningCurves(DFF, position['ry'], norm=True)		
		tuningcurve 	= smoothAngularTuningCurves(tuningcurve)			

		peaks 			= computeAngularPeaks(tuningcurve)
		si 				= computeSpatialInfo(tuningcurve, position['ry'])		
		stat 			= computeRayleighTest(tuningcurve)	
		corr_tc			= computeCorrelationTC(DFF, position['ry'])
		pf, extent		= computePlaceFields(DFF, position[['x', 'z']], 15)

		SF[i] = A
		TC[i] = tuningcurve
		PF[i] = pf
		positions[i] = position
		allinfo[i] = pd.concat([peaks,si,stat,corr_tc], axis = 1)
		DFFS[i] = DFF
		Cs[i] = C

	return SF, TC, PF, allinfo, positions, DFFS, Cs


def loadLFP(path, n_channels=90, channel=64, frequency=1250.0, precision='int16'):
	import neuroseries as nts	
	f = open(path, 'rb')
	startoffile = f.seek(0, 0)
	endoffile = f.seek(0, 2)
	bytes_size = 2		
	n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
	duration = n_samples/frequency
	interval = 1/frequency
	f.close()
	fp = np.memmap(path, np.int16, 'r', shape = (n_samples, n_channels))		
	timestep = np.arange(0, n_samples)/frequency

	if type(channel) is not list:
		timestep = np.arange(0, n_samples)/frequency
		return nts.Tsd(timestep, fp[:,channel], time_units = 's')
	elif type(channel) is list:
		timestep = np.arange(0, n_samples)/frequency
		return nts.TsdFrame(timestep, fp[:,channel], time_units = 's')

def loadAuxiliary(path, n_probe = 1, fs = 20000):
	"""
	Extract the acceleration from the auxiliary.dat for each epochs
	Downsampled at 100 Hz
	Args:
		path: string
		epochs_ids: list        
	Return: 
		TsdArray
	""" 	
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	if 'Acceleration.h5' in os.listdir(os.path.join(path, 'Analysis')):
		accel_file = os.path.join(path, 'Analysis', 'Acceleration.h5')
		store = pd.HDFStore(accel_file, 'r')
		accel = store['acceleration'] 
		store.close()
		accel = nts.TsdFrame(t = accel.index.values*1e6, d = accel.values) 
		return accel
	else:
		aux_files = np.sort([f for f in os.listdir(path) if 'auxiliary' in f])
		if len(aux_files)==0:
			print("Could not find "+f+'_auxiliary.dat; Exiting ...')
			sys.exit()

		accel = []
		sample_size = []
		for i, f in enumerate(aux_files):
			new_path 	= os.path.join(path, f)
			f 			= open(new_path, 'rb')
			startoffile = f.seek(0, 0)
			endoffile 	= f.seek(0, 2)
			bytes_size 	= 2
			n_samples 	= int((endoffile-startoffile)/(3*n_probe)/bytes_size)
			duration 	= n_samples/fs		
			f.close()
			tmp 		= np.fromfile(open(new_path, 'rb'), np.uint16).reshape(n_samples,3*n_probe)
			accel.append(tmp)
			sample_size.append(n_samples)
			del tmp

		accel = np.concatenate(accel)	
		factor = 37.4e-6
		# timestep = np.arange(0, len(accel))/fs
		# accel = pd.DataFrame(index = timestep, data= accel*37.4e-6)
		tmp  = []
		for i in range(accel.shape[1]):
			tmp.append(scipy.signal.resample_poly(accel[:,i]*factor, 1, 100))
		tmp = np.vstack(tmp).T
		timestep = np.arange(0, len(tmp))/(fs/100)
		tmp = pd.DataFrame(index = timestep, data = tmp)
		accel_file = os.path.join(path, 'Analysis', 'Acceleration.h5')
		store = pd.HDFStore(accel_file, 'w')
		store['acceleration'] = tmp
		store.close()
		accel = nts.TsdFrame(t = tmp.index.values*1e6, d = tmp.values) 
		return accel		

def loadInfos(data_directory, sessions_dir, animals):
	infos = {}
	for fbasename in animals:		
		info = pd.read_csv(sessions_dir + 'datasets_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
		paths = [os.path.join(data_directory, fbasename[0:3] + '00', fbasename, fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
		sessions = [fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '') for i in info.index]
		info['paths'] = paths
		info['sessions'] = sessions
		info = info.set_index('sessions')
		infos[fbasename] = info

	return infos
