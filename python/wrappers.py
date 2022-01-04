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


def loadNeuropixel(path, index=None, fs = 20000):
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()    
	new_path = os.path.join(path, 'Analysis/')
	if os.path.exists(new_path):		
		files        = os.listdir(new_path)
		if 'SpikeData.mat' in files:
			spikedata     = scipy.io.loadmat(new_path+'SpikeData.mat')
			spikes         = {}
			n_neurons = len(spikedata['S'][0][0][0])
			for i in range(n_neurons):
				spikes[i]     = nts.Ts(spikedata['S'][0][0][0][i][0][0][0][1][0][0][2], time_units = 's')

	return spikes

def makeEpochs(path, order, file = None, start=None, end = None, time_units = 's'):
	"""
	The pre-processing pipeline should spit out a csv file containing all the successive epoch of sleep/wake
	This function will load the csv and write neuroseries.IntervalSet of wake and sleep in /Analysis/BehavEpochs.h5
	If no csv exists, it's still possible to give by hand the start and end of the epochs
	Notes:
		The function assumes no header on the csv file
	Args:
		path: string
		order: list
		file: string
		start: list/array (optional)
		end: list/array (optional)
		time_units: string (optional)
	Return: 
		none
	"""		
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()	
	if file:
		listdir 	= os.listdir(path)	
		if file not in listdir:
			print("The file "+file+" cannot be found in the path "+path)
			sys.exit()			
		filepath 	= os.path.join(path, file)
		epochs 		= pd.read_csv(filepath, header = None)
	elif file is None and len(start) and len(end):
		epochs = pd.DataFrame(np.vstack((start, end)).T)
	elif file is None and start is None and end is None:
		print("You have to specify either a file or arrays of start and end; Exiting ...")
		sys.exit()
	
	# Creating /Analysis/ Folder here if not already present
	new_path	= os.path.join(path, 'Analysis/')
	if not os.path.exists(new_path): os.makedirs(new_path)
	# Writing to BehavEpochs.h5
	new_file 	= os.path.join(new_path, 'BehavEpochs.h5')
	store 		= pd.HDFStore(new_file, 'a')
	epoch 		= np.unique(order)
	for i, n in enumerate(epoch):
		idx = np.where(np.array(order) == n)[0]
		ep = nts.IntervalSet(start = epochs.loc[idx,0],
							end = epochs.loc[idx,1],
							time_units = time_units)
		store[n] = pd.DataFrame(ep)
	store.close()

	return None

def makePositions_NeuroPixel(path, file_order, episodes, names = ['ry', 'rx', 'rz', 'x', 'y', 'z'], update_wake_epoch = True):
	"""
	Assuming that makeEpochs has been runned and a file BehavEpochs.h5 can be 
	found in /Analysis/, this function will look into path for analogin file 
	containing the TTL pulses. The position time for all events will thus be
	updated and saved in Analysis/Position.h5.
	BehavEpochs.h5 will although be updated to match the time between optitrack
	and intan
	
	Notes:
		The function assumes headers on the csv file of the position in the following order:
			['ry', 'rx', 'rz', 'x', 'y', 'z']
	Args:
		path: string
		file_order: list
		names: list
	Return: 
		None
	""" 
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	files = os.listdir(path)
	for f in file_order:
		if not np.any([f+'.csv' in g for g in files]):
			print("Could not find "+f+'.csv; Exiting ...')
			sys.exit()    
	new_path = os.path.join(path, 'Analysis/')
	if not os.path.exists(new_path): os.makedirs(new_path)                
	file_epoch = os.path.join(path, 'Analysis', 'BehavEpochs.h5')
	if os.path.exists(file_epoch):
		wake_ep = loadEpoch(path, 'wake')
	else:
		makeEpochs(path, episodes, file = 'Epoch_TS.csv')
		wake_ep = loadEpoch(path, 'wake')
	if len(wake_ep) != len(file_order):
		print("Number of wake episodes doesn't match; Exiting...")
		sys.exit()

	print(path)
	print(file_order)

	# HACK FOR NEUROPIXEL
	# LOADING THE TTL FROM THE NIDQ DAT FILE
	file = os.path.join(path, [f for f in files if 'nidq.dat' in f][0])
	f = open(file, 'rb')
	startoffile = f.seek(0, 0)
	endoffile = f.seek(0, 2)
	bytes_size = 1
	n_samples = int((endoffile-startoffile)/2/bytes_size)	
	f.close()

	with open(file, 'rb') as f:
		data = np.fromfile(f, np.uint8).reshape((n_samples, 2))[:,0]

	data = data.astype(np.int32)

	peaks,_ = scipy.signal.find_peaks(np.diff(data), height=50)
	timestep = np.arange(0, len(data))/25000
	# analogin = pd.Series(index = timestep, data = data)
	peaks+=1
	ttl = pd.Series(index = timestep[peaks], data = data[peaks])    
	ttl = nts.Ts(t= ttl.index.values, time_units = 's')

	frames = []	

	for i, f in enumerate(file_order):		
		csv_file = os.path.join(path, "".join(s for s in files if f+'.csv' in s))
		position = pd.read_csv(csv_file, header = [4,5], index_col = 1)
		if 1 in position.columns:
			position = position.drop(labels = 1, axis = 1)
		position = position[~position.index.duplicated(keep='first')]

		ttle = ttl.restrict(wake_ep.loc[[i]])
					
		length = np.minimum(len(ttle), len(position))
		ttle = ttle.iloc[0:length]
		position = position.iloc[0:length]
				
		position.index = ttle.index
		print(position.index)
		wake_ep.iloc[i,0] = np.int64(np.maximum(wake_ep.iloc[i,0], position.index[0]))
		wake_ep.iloc[i,1] = np.int64(np.minimum(wake_ep.iloc[i,1], position.index[-1]))

		# putting columns in the right order 
		order = []
		for n in position.columns:
			if n[0] == 'Rotation':
				order.append('r'+n[1].lower())
			elif n[0] == 'Position':
				order.append(n[1].lower())
			else:
				print('Unknow csv file for position; Exiting')
				sys.exit()

		position.columns = order
		position = position[names]

		frames.append(position)


	
	position = pd.concat(frames)	
	position.columns = names
	position[['ry', 'rx', 'rz']] *= (np.pi/180)
	position[['ry', 'rx', 'rz']] += 2*np.pi
	position[['ry', 'rx', 'rz']] %= 2*np.pi

	position = nts.TsdFrame(position)
	

	if update_wake_epoch:
		store = pd.HDFStore(file_epoch, 'a')
		store['wake'] = pd.DataFrame(wake_ep)
		store.close()
	
	position_file = os.path.join(path, 'Analysis', 'Position.h5')
	store = pd.HDFStore(position_file, 'w')
	store['position'] = position.as_units('s')
	store.close()
	
	return

def loadEpoch(path, epoch, episodes = None):
	"""
	load the epoch contained in path	
	If the path contains a folder analysis, the function will load either the BehavEpochs.mat or the BehavEpochs.h5
	Run makeEpochs(data_directory, ['sleep', 'wake', 'sleep', 'wake'], file='Epoch_TS.csv') to create the BehavEpochs.h5

	Args:
		path: string
		epoch: string

	Returns:
		neuroseries.IntervalSet
	"""			
	if not os.path.exists(path): # Check for path
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	if epoch in ['sws', 'rem']: 		
		# loading the .epoch.evt file
		file = os.path.join(path,os.path.basename(path)+'.'+epoch+'.evt')
		if os.path.exists(file):
			tmp = np.genfromtxt(file)[:,0]
			tmp = tmp.reshape(len(tmp)//2,2)/1000
			ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
			# TO make sure it's only in sleep since using the TheStateEditor
			sleep_ep = loadEpoch(path, 'sleep')
			ep = sleep_ep.intersect(ep)
			return ep
		else:
			print("The file ", file, "does not exist; Exiting ...")
			sys.exit()
	elif epoch == 'wake.evt.theta':
		file = os.path.join(path,os.path.basename(path)+'.'+epoch)
		if os.path.exists(file):
			tmp = np.genfromtxt(file)[:,0]
			tmp = tmp.reshape(len(tmp)//2,2)/1000
			ep = nts.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
			return ep
		else:
			print("The file ", file, "does not exist; Exiting ...")
	filepath 	= os.path.join(path, 'Analysis')
	if os.path.exists(filepath): # Check for path/Analysis/	
		listdir		= os.listdir(filepath)
		file 		= [f for f in listdir if 'BehavEpochs' in f]
	if len(file) == 0: # Running makeEpochs		
		makeEpochs(path, episodes, file = 'Epoch_TS.csv')
		listdir		= os.listdir(filepath)
		file 		= [f for f in listdir if 'BehavEpochs' in f]
	if file[0] == 'BehavEpochs.h5':
		new_file = os.path.join(filepath, 'BehavEpochs.h5')
		store 		= pd.HDFStore(new_file, 'r')
		if '/'+epoch in store.keys():
			ep = store[epoch]
			store.close()
			return nts.IntervalSet(ep)
		else:
			print("The file BehavEpochs.h5 does not contain the key "+epoch+"; Exiting ...")
			sys.exit()
	elif file[0] == 'BehavEpochs.mat':
		behepochs = scipy.io.loadmat(os.path.join(filepath,file[0]))
		if epoch == 'wake':
			wake_ep = np.hstack([behepochs['wakeEp'][0][0][1],behepochs['wakeEp'][0][0][2]])
			return nts.IntervalSet(wake_ep[:,0], wake_ep[:,1], time_units = 's').drop_short_intervals(0.0)
		elif epoch == 'sleep':
			sleep_pre_ep, sleep_post_ep = [], []
			if 'sleepPreEp' in behepochs.keys():
				sleep_pre_ep = behepochs['sleepPreEp'][0][0]
				sleep_pre_ep = np.hstack([sleep_pre_ep[1],sleep_pre_ep[2]])
				sleep_pre_ep_index = behepochs['sleepPreEpIx'][0]
			if 'sleepPostEp' in behepochs.keys():
				sleep_post_ep = behepochs['sleepPostEp'][0][0]
				sleep_post_ep = np.hstack([sleep_post_ep[1],sleep_post_ep[2]])
				sleep_post_ep_index = behepochs['sleepPostEpIx'][0]
			if len(sleep_pre_ep) and len(sleep_post_ep):
				sleep_ep = np.vstack((sleep_pre_ep, sleep_post_ep))
			elif len(sleep_pre_ep):
				sleep_ep = sleep_pre_ep
			elif len(sleep_post_ep):
				sleep_ep = sleep_post_ep						
			return nts.IntervalSet(sleep_ep[:,0], sleep_ep[:,1], time_units = 's')
		###################################
		# WORKS ONLY FOR MATLAB FROM HERE #
		###################################		
		elif epoch == 'sws':
			sampling_freq = 1250
			new_listdir = os.listdir(path)
			for f in new_listdir:
				if 'sts.SWS' in f:
					sws = np.genfromtxt(os.path.join(path,f))/float(sampling_freq)
					return nts.IntervalSet.drop_short_intervals(nts.IntervalSet(sws[:,0], sws[:,1], time_units = 's'), 0.0)

				elif '-states.mat' in f:
					sws = scipy.io.loadmat(os.path.join(path,f))['states'][0]
					index = np.logical_or(sws == 2, sws == 3)*1.0
					index = index[1:] - index[0:-1]
					start = np.where(index == 1)[0]+1
					stop = np.where(index == -1)[0]
					return nts.IntervalSet.drop_short_intervals(nts.IntervalSet(start, stop, time_units = 's', expect_fix=True), 0.0)

		elif epoch == 'rem':
			sampling_freq = 1250
			new_listdir = os.listdir(path)
			for f in new_listdir:
				if 'sts.REM' in f:
					rem = np.genfromtxt(os.path.join(path,f))/float(sampling_freq)
					return nts.IntervalSet(rem[:,0], rem[:,1], time_units = 's').drop_short_intervals(0.0)

				elif '-states/m' in listdir:
					rem = scipy.io.loadmat(path+f)['states'][0]
					index = (rem == 5)*1.0
					index = index[1:] - index[0:-1]
					start = np.where(index == 1)[0]+1
					stop = np.where(index == -1)[0]
					return nts.IntervalSet(start, stop, time_units = 's', expect_fix=True).drop_short_intervals(0.0)

def loadPosition_NeuroPixel(path, events = None, episodes = None):
	"""
	load the position contained in /Analysis/Position.h5

	Notes:
		The order of the columns is assumed to be
			['ry', 'rx', 'rz', 'x', 'y', 'z']
	Args:
		path: string
		
	Returns:
		neuroseries.TsdFrame
	"""        
	if not os.path.exists(path): # Checking for path
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()
	new_path = os.path.join(path, 'Analysis')
	if not os.path.exists(new_path): os.mkdir(new_path)
	file = os.path.join(path, 'Analysis', 'Position.h5')
	if not os.path.exists(file):
		makePositions_NeuroPixel(path, events, episodes)
	if os.path.exists(file):
		store = pd.HDFStore(file, 'r')
		position = store['position']
		store.close()
		position = nts.TsdFrame(t = position.index.values, d = position.values, columns = position.columns, time_units = 's')
		return position
	else:
		print("Cannot find "+file+" for loading position")
		sys.exit()    	

