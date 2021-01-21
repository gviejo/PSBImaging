import sys,os
import neuroseries as nts
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
# from wrappers import *
# from functions import *
import sys
import scipy.signal
from pycircstat.descriptive import mean as circmean
from matplotlib.colors import hsv_to_rgb


def smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0):
	new_tuning_curves = {}	
	for i in tuning_curves.columns:
		tcurves = tuning_curves[i]
		offset = np.mean(np.diff(tcurves.index.values))
		padded 	= pd.Series(index = np.hstack((tcurves.index.values-(2*np.pi)-offset,
												tcurves.index.values,
												tcurves.index.values+(2*np.pi)+offset)),
							data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
		smoothed = padded.rolling(window=window,win_type='gaussian',center=True,min_periods=1).mean(std=deviation)		
		new_tuning_curves[i] = smoothed.loc[tcurves.index]

	new_tuning_curves = pd.DataFrame.from_dict(new_tuning_curves)

	return new_tuning_curves

def findHDCells(tuning_curves, z = 50, p = 0.0001 , m = 0):
	"""
		Peak firing rate larger than 1
		and Rayleigh test p<0.001 & z > 100
	"""
	cond1 = tuning_curves.max()>m
	from pycircstat.tests import rayleigh
	stat = pd.DataFrame(index = tuning_curves.columns, columns = ['pval', 'z'])
	for k in tuning_curves:
		stat.loc[k] = rayleigh(tuning_curves[k].index.values, tuning_curves[k].values)
	cond2 = np.logical_and(stat['pval']<p,stat['z']>z)
	tokeep = stat.index.values[np.where(np.logical_and(cond1, cond2))[0]]
	return tokeep, stat




path = '/mnt/DataAdrienBig/PeyracheLabData/Guillaume/A0600/A0634/A0634-201124'
name = path.split('/')[-1]
 

#############################################################################################################
# LOADING CALCIUM TRANSIENTS
#############################################################################################################
C = pd.read_csv(path+'/'+name+'_C.csv', header = None)
C = C.T

# DF / F
# DF = C.diff()
# DF.loc[0] = DF.loc[1]
# # DFF = DF/C
# C = DF
# C = DFF

#############################################################################################################
# LOADING THE ANGLE
#############################################################################################################
csv_file = path + '/' + name + '_0.csv'
position = pd.read_csv(csv_file, header = [4,5], index_col = 1)
if 1 in position.columns:
	position = position.drop(labels = 1, axis = 1)
position = position[~position.index.duplicated(keep='first')]
position.columns = ['ry', 'rx', 'rz', 'x', 'y', 'z']
position[['ry', 'rx', 'rz']] *= (np.pi/180)
position[['ry', 'rx', 'rz']] += 2*np.pi
position[['ry', 'rx', 'rz']] %= 2*np.pi



#############################################################################################################
# LOADING THE ANALOGIN
#############################################################################################################
analogin_file = path + '/' + name + '_0_analogin.dat'
f = open(analogin_file, 'rb')
startoffile = f.seek(0, 0)
endoffile = f.seek(0, 2)
bytes_size = 2        
n_samples = int((endoffile-startoffile)/2/bytes_size)
f.close()
n_channels = 2
with open(analogin_file, 'rb') as f:
	analogin = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
analogin = analogin.astype(np.int32)
peaks,_ = scipy.signal.find_peaks(np.diff(analogin[:,0]), height=30000)
timestep = np.arange(0, len(analogin))/20000
peaks+=1
ttl_tracking = pd.Series(index = timestep[peaks], data = analogin[peaks,0])

peaks,_ = scipy.signal.find_peaks(np.abs(np.diff(analogin[:,1])), height=30000, distance = 500)
ttl_miniscope = pd.Series(index = timestep[peaks], data = analogin[peaks,1])

#############################################################################################################
# ALIGNING
#############################################################################################################
start_track = ttl_tracking.index.values[0]
end_track = ttl_tracking.index.values[-1]

start_mini = ttl_miniscope.index.values[0]

time_frame = ttl_miniscope.index.values[0:np.minimum(len(ttl_miniscope),len(C))]

start_end = (np.argmin(np.abs(time_frame - start_track)),
			np.argmin(np.abs(time_frame - end_track))
				)

time_frame = time_frame[start_end[0]:start_end[1]]

C = C[start_end[0]:start_end[1]]

angle = pd.Series(index = position['ry'].index.values + start_track, data = position['ry'].values)


tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))	
tmp2 			= tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)	


time_bins		= np.zeros(len(time_frame)+1)
time_bins[1:-1] = time_frame[1:] - np.diff(time_frame)/2
time_bins[0] = time_frame[0] - (1/30)
time_bins[-1] = time_frame[-1] + (1/30)

index 			= np.digitize(tmp2.index.values, time_bins)
tmp3 			= tmp2.groupby(index).mean()
tmp3.index 		= time_bins[np.unique(index)-1]
newangle 		= pd.Series(index = tmp3.index.values, data = tmp3.values%(2*np.pi))

newangle = newangle.iloc[0:-1]

#############################################################################################################
# MAKING TUNING CURVES
#############################################################################################################
ang_bins = np.linspace(0, 2*np.pi, 121)

idx = np.digitize(newangle, ang_bins)-1

tc = np.zeros((120,C.shape[1]))

for i in range(tc.shape[0]):
	tc[i,:] = C[idx==i].mean(0)

tc = pd.DataFrame(index = ang_bins[0:-1] + np.diff(ang_bins)/2, data = tc)
tc = tc.rolling(window=10,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)

figure()
for i in range(C.shape[1]):
	subplot(10,5,i+1,projection='polar')
	plot(tc[i])
	title(i)
	# xticks([])
	# yticks([])





DFF = C.diff()
DFF = DFF.fillna(0)
DFF[DFF<0] = 0

# import neuroseries as nts
# spikes = {}
# for i in tokeep:
# 	tmp = scipy.signal.find_peaks(DFF[i].values, height=0.0001)[0]
# 	spikes[i] = nts.Ts(t = time_frame[tmp], time_units = 's')

# wake_ep = nts.IntervalSet(start = time_frame[0], end = time_frame[-1], time_units = 's')

# angle = nts.Tsd(t = angle.index.values, d = angle.values, time_units = 's')

# from functions import *

# tc3 = computeAngularTuningCurves(spikes, angle, wake_ep, 61)
# tc3 = smoothAngularTuningCurves(tc3)

tc2 = np.zeros((120,DFF.shape[1]))

for i in range(tc2.shape[0]):
	tc2[i,:] = DFF[idx==i].mean(0)

tc2 = pd.DataFrame(index = ang_bins[0:-1] + np.diff(ang_bins)/2, data = tc2)
tc2 = smoothAngularTuningCurves(tc2)


# tc3.columns = tc2.columns

figure()
for i,n in enumerate(tc2.columns):
	ax = subplot(10,5,i+1, projection = 'polar')
	plot(tc2[i])
	# ax2 = ax.twinx()
	# plot(tc3[i])
	title(i)
	xticks([])
	yticks([])


path = '/mnt/DataAdrienBig/PeyracheLabData/Guillaume/A0600/A0634/A0634-201124/A0634-201124_A.csv'


A = pd.read_csv(path, header = None)
A = A.values

dims = (304,304)


af = np.zeros((A.shape[1], dims[0], dims[1]))

for i in range(A.shape[1]):
	af[i] = A[:,i].reshape(dims)

# af = af[tokeep]

# tokeep, stat 						= findHDCells(tc2, z=10, p = 0.001)

peaks 								= pd.Series(index=tc2.columns,data = np.array([circmean(tc2.index.values, tc2[i].values) for i in tc2.columns]))

H = peaks.values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)

colorA = np.zeros((dims[0], dims[1], 3))
colorA *= np.nan

for i in range(len(af)):
	colorA[af[i].T > 3] = RGB[i]

figure()
imshow(colorA)

tokeep = [0, 1, 2, 5, 8, 10, 11, 13, 14, 16, 17, 18, 20, 25, 30, 31, 34, 35, 36, 37, 40, 45]

figure()
for i,n in enumerate(tokeep):
	ax = subplot(5,6,i+1, projection = 'polar')
	plot(tc2[n])
	# ax2 = ax.twinx()
	# plot(tc3[i])
	title(n)
	# xticks([])
	# yticks([])



wakangle = newangle
H = wakangle.values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)

# RGB = np.array([hsluv.hsluv_to_rgb(HSV[i]) for i in range(len(HSV))])
# sys.exit()

# sys.exit()

# resampling at 5 Hz
time_bins = np.arange(time_frame[0], (time_frame[-1] - time_frame[0]), 0.3)
idx = np.digitize(time_frame, time_bins)

wakangle5hz = wakangle.groupby(idx).mean()
DFF5hz = DFF.groupby(idx).mean()




from sklearn.manifold import Isomap

tmp = DFF5hz[tokeep].rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=5).values
H = wakangle5hz.values/(2*np.pi)
HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
RGB = hsv_to_rgb(HSV)

idx = tmp.sum(1)>0.055

tmp = tmp[idx]
RGB = RGB[idx]

imap = Isomap(n_neighbors = 100, n_components = 2).fit_transform(tmp)
figure()
subplot(122)
scatter(imap[:,0], imap[:,1], c= RGB, marker = '.', alpha = 0.4, linewidth = 0, s = 150)

subplot(121)

peaks 								= pd.Series(index=tokeep,data = np.array([circmean(tc2[tokeep].index.values, tc2[i].values) for i in tokeep]))

H2 = peaks.values/(2*np.pi)
HSV2 = np.vstack((H2, np.ones_like(H2), np.ones_like(H2))).T
RGB2 = hsv_to_rgb(HSV2)

colorA = np.ones((dims[0], dims[1], 3))
# colorA *= np.nan

for i, n in enumerate(tokeep):
	colorA[af[n].T > 3] = RGB2[i]

nonhd = list(set(np.arange(len(af))) - set(tokeep))

for i, n in enumerate(nonhd):
	colorA[af[n].T > 4] = [0,0,0]


imshow(colorA[50:200,75:250])





# from umap import UMAP

# ump = UMAP(n_neighbors = 100, min_dist = 1).fit_transform(tmp)
# figure()
# scatter(ump[:,0], ump[:,1], c= RGB, marker = 'o', alpha = 0.5, linewidth = 0, s = 100)

# show()



# from sklearn.decomposition import PCA
# pc = PCA(n_components = 2).fit_transform(tmp)

# figure()
# scatter(pc[:,0], pc[:,1], c= RGB, marker = '.', alpha = 0.5, linewidth = 0, s = 100)