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

data_directory = '/mnt/Data2/PSB/A8603/A8603-210602'

#episodes = ['sleep', 'wake']
episodes = ['sleep', 'wake', 'sleep', 'wake', 'sleep']

events = ['1', '3']



spikes 				= loadNeuropixel(data_directory)

position 			= loadPosition_NeuroPixel(data_directory, events, episodes)
wake_ep 			= loadEpoch(data_directory, 'wake', episodes)
sleep_ep 			= loadEpoch(data_directory, 'sleep')					


tuning_curves 	= computeAngularTuningCurves(spikes, position['ry'], wake_ep.loc[[0]], 60)

tokeep, stat 	= findHDCells(tuning_curves, z = 10, p = 0.0001 , m = 2)


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