import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
from matplotlib import gridspec
import sys
from scipy.ndimage.filters import gaussian_filter
from matplotlib.colors import hsv_to_rgb
from umap import UMAP
from sklearn.manifold import Isomap


def plotTuningCurves(tcurves, tokeep = []):	
	figure()
	for i in tcurves.columns:
		subplot(int(np.ceil(np.sqrt(tcurves.shape[1]))),int(np.ceil(np.sqrt(tcurves.shape[1]))),i+1, projection='polar')
		plot(tcurves[i])
		xticks([])
		yticks([])
	return

data_directory = '/media/guillaume/Elements/A8607-220102'


#episodes = ['sleep', 'wake', 'sleep', 'wake', 'sleep']
#events = ['1', '3']
episodes = ['sleep', 'wake']
events = ['1']



spikes 				= loadNeuropixel(data_directory)

position 			= loadPosition_NeuroPixel(data_directory, events, episodes)
wake_ep 			= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 			= loadEpoch(data_directory, 'sleep')					

#spikes = {n:spikes[n] for n in np.arange(0, 83)}

tuning_curves 	= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 60)
#tuning_curves2 	= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[1]], 60)

tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 2.0)

tokeep, stat 	= findHDCells(tuning_curves, z = 10, p = 0.0001 , m = 2)


plotTuningCurves(tuning_curves)

sys.exit()

#tokeep = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 18, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 35, 37, 38, 40, 42, 43, 44, 45, 46, 48, 50, 52, 53, 56, 57, 58, 60, 62, 64, 65, 67, 69, 70, 72, 74, 75, 76, 80, 81])

mean_fr = computeMeanFiringRate(spikes, [wake_ep.loc[[i]] for i in wake_ep.index], ['wake1', 'wake2'])
mean_fr = mean_fr.loc[tokeep]
peak_fr = pd.concat([tuning_curves[tokeep].max(), tuning_curves2[tokeep].max()], 1)

figure()
subplot(231)
plot(mean_fr['wake1'], mean_fr['wake2'], 'o')
subplot(232)
plot((mean_fr['wake2'] - mean_fr['wake1']).sort_values().values, '.-')
subplot(233)
[plot(np.arange(2), np.log(mean_fr.loc[i].values)) for i in mean_fr.index]

subplot(234)
plot(peak_fr[0], peak_fr[1], 'o')
subplot(235)
plot((peak_fr[1] - peak_fr[0]).sort_values().values, '.-')
subplot(236)
[plot(np.arange(2), np.log(peak_fr.loc[i].values)) for i in mean_fr.index]

# Dropping neurons with a difference of peak fireing rate larger than 10 Hz

tokeep = tokeep[np.abs(peak_fr[1] - peak_fr[0])<10]
#tokeep = tokeep[np.abs(mean_fr['wake1'] - mean_fr['wake2'])<2]

print(len(tokeep))




show()

sys.exit()
#############################################################
# CROSS CORRS
#############################################################
cc1 = compute_CrossCorrs({n:spikes[n] for n in tokeep}, wake_ep.loc[[0]], binsize=5, nbins = 200, norm = True)
cc2 = compute_CrossCorrs({n:spikes[n] for n in tokeep}, wake_ep.loc[[1]], binsize=5, nbins = 200, norm = True)
cc1 = cc1.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)
cc2 = cc2.rolling(window=20, win_type='gaussian', center = True, min_periods = 1).mean(std = 2.0)


tcurves = tuning_curves[tokeep]
peaks = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		

new_index = cc1.columns
pairs = pd.Series(index = new_index)
for i,j in pairs.index:	
	a = peaks[i] - peaks[j]
	pairs[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))

pairs = pairs.dropna().sort_values()

figure()
for i, cc in enumerate([cc1, cc2]):
	subplot(1,2,i+1)
	imshow(cc[pairs.index].T, aspect = 'auto', cmap = 'jet', interpolation = 'bilinear')
	xticks([0, np.where(cc.index.values == 0)[0][0], len(cc)], [cc.index[0], 0, cc.index[-1]])

show()

tcurves = tuning_curves2[tokeep]
peaks2 = pd.Series(index=tcurves.columns,data = np.array([circmean(tcurves.index.values, tcurves[i].values) for i in tcurves.columns])).sort_values()		

pairs2 = pd.Series(index = new_index)
for i,j in pairs2.index:	
	a = peaks2[i] - peaks2[j]
	pairs2[(i,j)] = np.minimum(np.abs(a), 2*np.pi - np.abs(a))




#############################################################
# UMAP
#############################################################
# binning spike train
bin_size = 300

data = []
sessions = []
angles = []
for e in [0,1]:
	ep = wake_ep.loc[[e]]
	bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)

	spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = tokeep)
	for i in tokeep:
		spks = spikes[i].as_units('ms').index.values
		spike_counts[i], _ = np.histogram(spks, bins)

	rate = np.sqrt(spike_counts/(bin_size*1e-3))
	#rate = spike_counts/(bin_size*1e-3)

	rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=3)

	rate = nts.TsdFrame(t = rate.index.values, d = rate.values, time_units = 'ms')

	#new_ep = refineWakeFromAngularSpeed(position['ry'], ep, bin_size = 300, thr = 0.1)

	#rate = rate.restrict(new_ep)

	angle = position['ry'].restrict(ep)
	wakangle = pd.Series(index = np.arange(len(bins)-1), dtype = np.float)
	tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
	wakangle.loc[tmp.index] = tmp
	wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)
	wakangle = nts.Tsd(t = wakangle.index.values, d = wakangle.values, time_units = 'ms')
	#wakangle = wakangle.restrict(new_ep)

	H = wakangle.values/(2*np.pi)
	HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
	RGB = hsv_to_rgb(HSV)

	# # cutting the 20th percentile
	# tmp = rate.values
	# index = tmp.sum(1) > np.percentile(tmp.sum(1), 20)

	# tmp = tmp[index,:]
#	RGB = RGB[index,:]

	
	# ump = UMAP(n_neighbors = 500, min_dist = 1).fit_transform(tmp)
	# scatter(ump[:,0], ump[:,1], c = RGB)
	# show()
	# break


	data.append(rate)
	sessions.append(np.ones(len(rate))*e)
	angles.append(RGB)


#ump = UMAP(n_neighbors = 500, min_dist = 1).fit_transform(data[1].values)
#scatter(ump[:,0], ump[:,1], c = angles[1])
#show()


data = pd.concat(data)
sessions = np.hstack(sessions)
angles = np.vstack(angles)


#ump = UMAP(n_neighbors = 400, min_dist = 0.1).fit_transform(data[sessions==0].values)

ump = Isomap(n_components = 3, n_neighbors = 50).fit_transform(data.values)

# scatter(ump[:,0], ump[:,1], c = angles[sessions==0],  alpha = 0.5, linewidth = 0, s = 40)
# show()


from mpl_toolkits.mplot3d import Axes3D


fig = figure()
mkrs = ['o', 's']
colors = ['blue', 'green']
ax = fig.add_subplot(121, projection='3d')
for i, n in enumerate(np.unique(sessions)):
	ax.scatter(ump[sessions==n,0], ump[sessions==n,1], ump[sessions==n,2], c= angles[sessions==n], marker = mkrs[i], alpha = 0.5, linewidth = 0, s = 50)

ax = fig.add_subplot(122, projection='3d')
for i, n in enumerate(np.unique(sessions)):
	ax.scatter(ump[sessions==n,0], ump[sessions==n,1], ump[sessions==n,2], color = colors[i])


show()











