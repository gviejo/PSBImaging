import numpy as np
from numba import jit
import pandas as pd
import neuroseries as nts
import sys, os
import scipy
from scipy import signal
from itertools import combinations
from pycircstat.descriptive import mean as circmean
from pylab import *

'''
Utilities functions
Feel free to add your own
'''

def computeCalciumTuningCurves(C, angle, nb_bins=120, norm = True):
	'''
		Downsampling the angle to the time bins based on C time index
	'''
	time_frame 		= C.index.values
	time_bins		= np.zeros(len(time_frame)+1)
	time_bins[1:-1] = time_frame[1:] - np.diff(time_frame)/2
	time_bins[0] = time_frame[0] - np.mean(np.diff(time_frame))
	time_bins[-1] = time_frame[-1] + np.mean(np.diff(time_frame))

	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))	
	tmp2 			= tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
	index 			= np.digitize(tmp2.index.values, time_bins)
	tmp3 			= tmp2.groupby(index).mean()	
	if 0 in tmp3.index: tmp3 			= tmp3.drop(0)
	if len(time_bins) in tmp3.index: tmp3			= tmp3.drop(len(time_bins))

	tmp3.index 		= time_bins[0:-1] + np.diff(time_bins)/2
	newangle 		= pd.Series(index = tmp3.index.values, data = tmp3.values%(2*np.pi))

	assert len(newangle) == len(C), "len(C) != len(newangle)"

	ang_bins = np.linspace(0, 2*np.pi, nb_bins+1)
	idx = np.digitize(newangle, ang_bins)-1
	tc = np.zeros((120,C.shape[1]))
	for i in range(tc.shape[0]):
		tc[i,:] = C[idx==i].mean(0)

	tc = pd.DataFrame(index = ang_bins[0:-1] + np.diff(ang_bins)/2, data = tc)

	# tc = tc - tc.min()
	# tc = tc / tc.max()
	tc = tc / tc.sum()

	return tc


def smoothAngularTuningCurves(tuning_curves, window = 30, deviation = 4.0):
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
	#tokeep = stat[stat['z']>p].index.values
	return tokeep, stat

def computeSpatialInfo(tc, angle):
	nb_bins = tc.shape[0]+1
	bins 	= np.linspace(0, 2*np.pi, nb_bins)	
	# Smoothing the angle here
	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
	tmp2 			= tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
	angle			= nts.Tsd(tmp2%(2*np.pi))
	pf = tc.values
	occupancy, _ 	= np.histogram(angle, bins)
	occ = np.atleast_2d(occupancy/occupancy.sum()).T
	f = np.sum(pf * occ, 0)
	pf = pf / f
	SI = np.sum(occ * pf * np.log2(pf), 0)
	SI = pd.DataFrame(index = tc.columns, columns = ['SI'], data = SI)
	SI = SI.fillna(0)
	return SI

def computeRayleighTest(tc):
	from pycircstat.tests import rayleigh
	stat = pd.DataFrame(index = tc.columns, columns = ['pval', 'z'])
	for k in tc:
		stat.loc[k] = rayleigh(tc[k].index.values, tc[k].values)
	return stat

def correlateTC(tc, pairs=None):
	cc_tc = []
	if pairs is None:
		pairs = list(combinations(tc.columns,2))
	for p in pairs:
		tmp = tc[list(p)].values		
		tmp2 = np.tile(tmp, (3,1))
		tmp3 = np.correlate(tmp2[:,0], tmp2[:,1], 'same').reshape(3,len(tmp)).T[:,1]
		tmp3 = pd.DataFrame(index = tc.index.values-np.pi, data = tmp3, columns = [tuple(p)])
		cc_tc.append(tmp3)
	cc_tc = pd.concat(cc_tc, 1)		
	return cc_tc

def computePlaceFields(C, position, nb_bins = 20):
	time_frame 		= C.index.values
	time_bins		= np.zeros(len(time_frame)+1)
	time_bins[1:-1] = time_frame[1:] - np.diff(time_frame)/2
	time_bins[0] = time_frame[0] - np.mean(np.diff(time_frame))
	time_bins[-1] = time_frame[-1] + np.mean(np.diff(time_frame))

	index 			= np.digitize(position.index.values, time_bins)
	tmp 			= position.groupby(index).mean()	
	if 0 in tmp.index: 	tmp 			= tmp.drop(0)
	if len(time_bins) in tmp.index: tmp			= tmp.drop(len(time_bins))	

	xpos = tmp.iloc[:,0]
	ypos = tmp.iloc[:,1]
	xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
	ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)
	xidx = np.digitize(xpos, xbins)-1
	yidx = np.digitize(ypos, ybins)-1

	place_fields = np.zeros((C.shape[1], nb_bins, nb_bins))
	for i in range(nb_bins):
		for j in range(nb_bins):
			idx = np.logical_and(xidx==i, yidx==j)
			if np.sum(idx):
				place_fields[:,i,j] = C[idx].mean(0)
		
	extent = (xbins[0], xbins[-1], ybins[0], ybins[-1]) # USEFUL FOR MATPLOTLIB
	return place_fields, extent

def computePairwiseAngularDifference(alltc, sessions):
	"""
	"""
	diffpairs = pd.DataFrame(index = list(combinations(alltc.keys(),2)), columns = sessions, dtype = np.float32)

	for i in diffpairs.columns:
		tmp = []
		columns = []
		for n in alltc.keys():
			if alltc[n][i].sum():
				tmp.append(alltc[n][i])
				columns.append(n)
		tmp = pd.concat(tmp, 1)
		tmp.columns = columns
		cc = correlateTC(tmp)
		diff = cc.idxmax().abs()	
		diffpairs.loc[diff.index.values,i] = diff.values

	diffsess = pd.DataFrame(index = diffpairs.index.values, columns = list(combinations(sessions, 2)), dtype = np.float32)

	for i, j in diffsess.columns:
		d = diffpairs[i] - diffpairs[j]
		diffsess[(i,j)] = np.abs(np.arctan2(np.sin(d.values.astype(np.float32)), np.cos(d.values.astype(np.float32))))

	return diffsess

def computePeaksAngularDifference(alltc, sessions):
	"""
	"""
	cc_tc_neurons = {}
	diffpeaks = pd.DataFrame(index = alltc.keys(), columns = list(combinations(sessions,2)))

	for n in alltc.keys():
		tmp = alltc[n][sessions].dropna(1)
		if tmp.shape[1] > 1:
			cc = correlateTC(tmp)
			cc_tc_neurons[n] = cc
			tmp = cc.idxmax().abs()	
			diffpeaks.loc[n,tmp.index.values] = tmp.values

	return diffpeaks

def centerTuningCurves(tcurve):
	"""
	center tuning curves by peak
	"""
	peak 			= pd.Series(index=tcurve.columns,data = np.array([circmean(tcurve.index.values, tcurve[i].values) for i in tcurve.columns]))
	new_tcurve 		= []
	for p in tcurve.columns:	
		x = tcurve[p].index.values - tcurve[p].index[tcurve[p].index.get_loc(peak[p], method='nearest')]
		x[x<-np.pi] += 2*np.pi
		x[x>np.pi] -= 2*np.pi
		tmp = pd.Series(index = x, data = tcurve[p].values).sort_index()
		new_tcurve.append(tmp.values)
	new_tcurve = pd.DataFrame(index = np.linspace(-np.pi, np.pi, tcurve.shape[0]+1)[0:-1], data = np.array(new_tcurve).T, columns = tcurve.columns)
	return new_tcurve


def checkTuningCurvesCrossCorr(cc, alltc, session, i):
	p = cc.columns[np.random.randint(cc.shape[1])]
	figure(figsize = (13,8))
	subplot(131,projection = 'polar')
	plot(alltc[p[0]][session])
	plot(alltc[p[1]][session], '--')
	subplot(132)
	plot(alltc[p[0]][session])
	plot(alltc[p[1]][session], '--')	
	subplot(133)
	plot(cc[p])
	show(block = True)


def computeAngularPeaks(tuningcurve):
	peaks 			= pd.DataFrame(index=tuningcurve.keys(),
		data = np.array([circmean(tuningcurve[i].index.values, tuningcurve[i].values) for i in tuningcurve.keys()]),
		columns = ['peaks']
		)	
	return peaks

def computeCorrelationTC(DFF, angle):
	tcurves2 = []
	DFF2 = np.array_split(DFF,2)	
	for j in range(2):		
		tcurves_half	= computeCalciumTuningCurves(DFF2[j], angle, norm=True)
		tcurves_half 	= smoothAngularTuningCurves(tcurves_half)	
		tcurves2.append(tcurves_half)

	diff = [np.corrcoef(tcurves2[0][j], tcurves2[1][j])[0,1] for j in DFF.columns]
	diff = pd.DataFrame(data = diff, columns = ['halfcorr'])
	return diff

# Selecting neurons with good tuning curves
from scipy import optimize
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

def findSinglePeakHDCell(alltc, sessions):
	'''
	FITTING NORMAL DISTRIBUTION TO TC
	'''
	std = pd.DataFrame(index = list(alltc.keys()), columns = sessions, dtype = np.float32)

	for n in alltc.keys():
		tmp = alltc[n]
		ctc = centerTuningCurves(tmp.dropna(1))	
		for k in ctc.columns:
			try:
				popt, _ = optimize.curve_fit(gaussian, ctc.index.values, ctc[k].values)
			except RuntimeError:
				popt = np.ones(3)*np.nan
			std.loc[n,k] = popt[-1]
	return std
	
def downsampleAngleFromC(time_frame, angle):
	'''
		Downsampling the angle to the time bins based on C time index
	'''	
	time_bins		= np.zeros(len(time_frame)+1)
	time_bins[1:-1] = time_frame[1:] - np.diff(time_frame)/2
	time_bins[0] = time_frame[0] - np.mean(np.diff(time_frame))
	time_bins[-1] = time_frame[-1] + np.mean(np.diff(time_frame))

	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))	
	tmp2 			= tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
	index 			= np.digitize(tmp2.index.values, time_bins)
	tmp3 			= tmp2.groupby(index).mean()	
	if 0 in tmp3.index: tmp3 			= tmp3.drop(0)
	if len(time_bins) in tmp3.index: tmp3			= tmp3.drop(len(time_bins))

	tmp3.index 		= time_bins[0:-1] + np.diff(time_bins)/2
	newangle 		= pd.Series(index = tmp3.index.values, data = tmp3.values%(2*np.pi))

	return newangle



###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.4          # height in inches
	fig_size = [fig_width,fig_height]
	return fig_size

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)

def noaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.set_xticks([])
	ax.set_yticks([])
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)

def getColoredFootprints(A, peaks, thr, nb_bins = 5):
	from matplotlib.colors import hsv_to_rgb
	dims = A.shape[1:]
	H = peaks.values/(2*np.pi)
	# binning angles
	idx = np.digitize(H, np.linspace(0, 1, nb_bins+1))-1
	H = idx/(nb_bins-1)
	HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
	RGB = hsv_to_rgb(HSV)

	colorA = np.zeros((dims[0], dims[1], 3))
	colorA *= np.nan

	for i in range(len(A)):
		colorA[A[i] > thr] = RGB[i]

	return colorA	


def computeAngularTuningCurves(spikes, angle, ep, nb_bins = 180, frequency = 120.0):
	bins 			= np.linspace(0, 2*np.pi, nb_bins)
	idx 			= bins[0:-1]+np.diff(bins)/2
	tuning_curves 	= pd.DataFrame(index = idx, columns = list(spikes.keys()))	
	angle 			= angle.restrict(ep)
	# Smoothing the angle here
	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
	tmp2 			= tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
	angle			= nts.Tsd(tmp2%(2*np.pi))
	for k in spikes:
		spks 			= spikes[k]
		# true_ep 		= nts.IntervalSet(start = np.maximum(angle.index[0], spks.index[0]), end = np.minimum(angle.index[-1], spks.index[-1]))		
		spks 			= spks.restrict(ep)	
		angle_spike 	= angle.restrict(ep).realign(spks)
		spike_count, bin_edges = np.histogram(angle_spike, bins)
		occupancy, _ 	= np.histogram(angle, bins)
		spike_count 	= spike_count/occupancy		
		tuning_curves[k] = spike_count*frequency	

	return tuning_curves

def computeMeanFiringRate(spikes, epochs, name):
	mean_frate = pd.DataFrame(index = spikes.keys(), columns = name, dtype = np.float32)
	for n, ep in zip(name, epochs):
		for k in spikes:
			mean_frate.loc[k,n] = len(spikes[k].restrict(ep))/ep.tot_length('s')
	return mean_frate


#########################################################
# CORRELATION
#########################################################
@jit(nopython=True)
def crossCorr(t1, t2, binsize, nbins):
	''' 
		Fast crossCorr 
	'''
	nt1 = len(t1)
	nt2 = len(t2)
	if np.floor(nbins/2)*2 == nbins:
		nbins = nbins+1

	m = -binsize*((nbins+1)/2)
	B = np.zeros(nbins)
	for j in range(nbins):
		B[j] = m+j*binsize

	w = ((nbins/2) * binsize)
	C = np.zeros(nbins)
	i2 = 1

	for i1 in range(nt1):
		lbound = t1[i1] - w
		while i2 < nt2 and t2[i2] < lbound:
			i2 = i2+1
		while i2 > 1 and t2[i2-1] > lbound:
			i2 = i2-1

		rbound = lbound
		l = i2
		for j in range(nbins):
			k = 0
			rbound = rbound+binsize
			while l < nt2 and t2[l] < rbound:
				l = l+1
				k = k+1

			C[j] += k

	# for j in range(nbins):
	# C[j] = C[j] / (nt1 * binsize)
	C = C/(nt1 * binsize/1000)

	return C


def compute_CrossCorrs(spks, ep, binsize=10, nbins = 2000, norm = False):
	"""
		
	"""	
	neurons = list(spks.keys())
	times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
	cc = pd.DataFrame(index = times, columns = list(combinations(neurons, 2)))
		
	for i,j in cc.columns:		
		spk1 = spks[i].restrict(ep).as_units('ms').index.values
		spk2 = spks[j].restrict(ep).as_units('ms').index.values		
		tmp = crossCorr(spk1, spk2, binsize, nbins)		
		fr = len(spk2)/ep.tot_length('s')
		if norm:
			cc[(i,j)] = tmp/fr
		else:
			cc[(i,j)] = tmp
	return cc

def plotTuningCurves(tcurves, tcurves2, tokeep):
	figure()
	for i in range(len(tcurves.columns)):	
		subplot(int(np.ceil(np.sqrt(tcurves.shape[1]))),int(np.ceil(np.sqrt(tcurves.shape[1]))),i+1, projection='polar')		
		plot(tcurves.iloc[:,i])
		plot(tcurves2.iloc[:,i])
		if tcurves.columns[i] in tokeep:
			plot(tcurves.iloc[:,i], linewidth = 3)
			plot(tcurves2.iloc[:,i], linewidth = 3)
		xticks([0], [tcurves.columns[i]])
		yticks([])		
	show()
	return
