import numpy as np
from numba import jit
import pandas as pd
import neuroseries as nts
import sys, os
import scipy
from scipy import signal
from itertools import combinations
from pycircstat.descriptive import mean as circmean

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
	# cond2 = np.logical_and(stat['pval']<p,stat['z']>z)
	# tokeep = stat.index.values[np.where(np.logical_and(cond1, cond2))[0]]	
	tokeep = stat[stat['z']>p].index.values
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
	diffpairs = pd.DataFrame(index = list(combinations(alltc.keys(),2)), columns = sessions)

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

	diffsess = pd.DataFrame(index = diffpairs.index.values, columns = list(combinations(sessions, 2)))

	for i, j in diffsess.columns:
		d = diffpairs[i] - diffpairs[j]
		d = d.dropna()
		diffsess.loc[d.index.values,(i,j)] = np.abs(np.arctan2(np.sin(d.values.astype(np.float32)), np.cos(d.values.astype(np.float32))))

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


# for i in range(20):
# 	checkTuningCurvesCrossCorr(cc, alltc, 0, i)