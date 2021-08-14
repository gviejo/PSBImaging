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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import product, combinations
from scipy.stats import norm
from scipy.ndimage.filters import gaussian_filter

#dims = (304,304)
dims = (181,203)

data_directory = '/mnt/DataRAID/MINISCOPE'

fbasename = 'A6509'
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
paths = [os.path.join(data_directory, fbasename[0:3] + '00', fbasename, 'minian', fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
sessions = [fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '') for i in info.index]
info['paths'] = paths
info['sessions'] = sessions
info = info.set_index('sessions')

#############################################################################################
# Cell Registration
#############################################################################################
cellreg = pd.read_csv(os.path.join(data_directory, fbasename[0:3] + '00', fbasename, 'minian', 'minian_reg.csv'), index_col = [0])

# intersecting info and cellreg sessions
sessions = np.intersect1d(info.index.values, cellreg.columns.values)

info = info.loc[sessions]
cellreg = cellreg[sessions]

n_sessions_detected = cellreg.notna().sum(1)

SF = {}
TC = {}
HD = {}
PK = {}
SI = {}
RL = {}
PF = {}
stab_tc = {}

thr_si = 0.3
thr_rl = 2



for i, s in enumerate(sessions):
	path 			= info.loc[s,'paths']	
	if os.path.exists(path):
		print(path)
		name 			= os.path.basename(path)		
		A, C, position 	= loadCalciumData(path, dims = dims)
		
		
		DFF 			= C.diff()
		DFF 			= DFF.fillna(0).as_dataframe()
		DFF[DFF<0]		= 0

		tuningcurve		= computeCalciumTuningCurves(DFF, position['ry'], norm=True)
		tuningcurve 	= smoothAngularTuningCurves(tuningcurve)			
		peaks 			= pd.Series(index=tuningcurve.keys(),
			data = np.array([circmean(tuningcurve[i].index.values, tuningcurve[i].values) for i in tuningcurve.keys()]))	
		si 				= computeSpatialInfo(tuningcurve, position['ry'])		
		stat 			= computeRayleighTest(tuningcurve)	
		
		tcurves2 = []
		DFF2 = np.array_split(DFF,2)	
		for j in range(2):		
			tcurves_half	= computeCalciumTuningCurves(DFF2[j], position['ry'], norm=True)
			tcurves_half 	= smoothAngularTuningCurves(tcurves_half)	
			tcurves2.append(tcurves_half)

		# diff = np.sum(np.abs(tcurves2[0] - tcurves2[1]), 0)	
		diff = {}
		for j in tuningcurve.columns:
			diff[j] = np.corrcoef(tcurves2[0][j], tcurves2[1][j])[0,1]
		diff = pd.Series(diff)	

		pf, extent			= computePlaceFields(DFF, position[['x', 'z']], 15)
		

		# tokeep 			= si[(si>thr_si).values].index.values
		tokeep 			= stat[(stat['z']>thr_rl).values].index.values	
		SI[i] 			= si
		SF[i] 			= A
		TC[i] 			= tuningcurve
		HD[i] 			= tokeep
		PK[i] 			= peaks
		RL[i]			= stat
		stab_tc[i]	 	= diff
		PF[i] 			= pf




figure()
count = 1
for i in SF.keys():
	ax = subplot(int(np.sqrt(len(SF)))+1,int(np.sqrt(len(SF)))+1,count)
	# a = (SF[i]>np.percentile(SF[i],0.001)).sum(0)	
	imshow(SF[i].sum(0), cmap = 'viridis')
	count += 1


cellreg = cellreg.fillna(-1).astype(np.int).values

tokeep = np.where(n_sessions_detected > 1)[0]

allst = {}
for i in tokeep:
	allst[i] = pd.Series(index = np.arange(cellreg.shape[1]))
	for j in np.where(cellreg[i]!=-1)[0]:		
		allst[i][j] = stab_tc[j].loc[cellreg[i,j]]		

allst = pd.concat(allst, 1).T



# Selecting neurons with stable tuning curves
allst[allst<0.3] = np.nan

tokeep = allst[allst.notna().any(1)].index.values




alltc = {}
allsi = {}
allpf = {}
for i in tokeep:
	alltc[i] = pd.DataFrame(columns = sessions)
	allsi[i] = pd.Series(index = sessions)	
	allpf[i] = np.zeros((len(sessions), PF[0].shape[1], PF[0].shape[2],))
	for j in np.where(cellreg[i]!=-1)[0]:
		alltc[i][sessions[j]] = TC[j][cellreg[i,j]]
		allsi[i].loc[sessions[j]] = SI[j].loc[cellreg[i,j]].values
		allpf[i][j] = PF[j][cellreg[i,j]]

#####################################################################################
# PLOT ALL TUNING CURVES
#####################################################################################
index = np.array(list(alltc.keys()))[0:20]

rigs = ['Circular', 'Square', '8-arm maze', 'Open field']

figure()
for i, n in enumerate(index):	
	ax = subplot(int(np.ceil(np.sqrt(len(index))))-1,int(np.ceil(np.sqrt(len(index)))),i+1)
	gs = GridSpecFromSubplotSpec(1,len(rigs),ax)
	tmp = alltc[n].dropna(1, 'all')
	grp = info.loc[tmp.columns].groupby('Rig').groups
	for j, m in enumerate(rigs):
		if m in grp.keys():
			subplot(gs[0,j], projection = 'polar')
			plot(alltc[n][grp[m]], color = cm.rainbow(np.linspace(0,1,len(rigs)))[j])
			xticks([])
			yticks([])

sys.exit()

####################################################################################
# DIFF PEAKS + SESSIONS
####################################################################################
diffpeaks = computePeaksAngularDifference(alltc, sessions = sessions)
diffsess = computePairwiseAngularDifference(alltc, sessions = sessions )



####################################################################################
# RANDOMIZING 
####################################################################################
rnddiffsess = []
for k in range(20):
	print(k)
	rndcellreg = np.copy(cellreg[list(alltc.keys())])
	for t in range(rndcellreg.shape[1]):
		np.random.shuffle(rndcellreg[:,t])
	rndtc = {}
	for i in range(len(rndcellreg)):
		rndtc[i] = pd.DataFrame(columns = sessions)
		for j in np.where(rndcellreg[i]!=-1)[0]:
			rndtc[i][sessions[j]] = TC[j][rndcellreg[i,j]]

	rnddiffsess.append(computePairwiseAngularDifference(rndtc, sessions = sessions))

rnddiffsess = pd.concat(rnddiffsess)

#####################################################################################
# PLOT ALL TUNING CURVES
#####################################################################################
index = np.array(list(alltc.keys()))

# figure()
# gs = GridSpec(len(index),n_sessions_detected.max())
# for i,n in enumerate(index):
# 	for j in range(n_sessions_detected.max()):
# 		subplot(gs[i,j],projection = 'polar')
# 		plot(alltc[n][sessions[j]])
# 		xticks([])
# 		yticks([])

# figure()
# index = alltc.keys()
# for i, n in enumerate(index):
# 	subplot(int(np.ceil(np.sqrt(len(index)))),int(np.ceil(np.sqrt(len(index)))),i+1, projection='polar')
# 	for j in alltc[n].columns:
# 		plot(alltc[n][j])
# 	xticks([])
# 	yticks([])


# figure()
# index = list(alltc.keys())
# gs = GridSpec(int(np.ceil(np.sqrt(len(index)))),int(np.ceil(np.sqrt(len(index)))))
# count = 0
# for i in range(int(np.ceil(np.sqrt(len(index))))):
# 	for j in range(int(np.ceil(np.sqrt(len(index))))):
# 		gs2 = gs[i,j].subgridspec(1,2)
# 		n = index[count]
# 		subplot(gs2[0,0], projection='polar')
# 		for k in alltc[n].columns:
# 			plot(alltc[n][k])
# 		xticks([])
# 		yticks([])
# 		tmp = []		
# 		for k in range(len(allpf[n])):
# 			tmp.append(gaussian_filter(allpf[n][k], 2))
# 		tmp = np.array(tmp)
# 		gs3 = gs2[0,1].subgridspec(2,2)
# 		xp, yp = ([0,0,1,1],[0,1,0,1])
# 		for k in range(len(allpf[n])):
# 			subplot(gs3[xp[k],yp[k]])
# 			imshow(tmp[k], cmap = 'jet')#, vmin = tmp.min(), vmax = tmp.max())
# 			xticks([])
# 			yticks([])
# 		count += 1
# 		if count == len(index):
# 			break
# 	if count == len(index):
# 		break




#####################################################################################
# CELL REG ANGULAR DIFFERENCE OF PREFERRED DIRECTION STARTING AT FIRST SESSION
#####################################################################################
# index = tokeep[np.where(n_sessions_detected[tokeep] == len(SF))[0]]
envs = info.loc[sessions].groupby('Rig').groups

sinter = []
for a, b  in combinations(envs.keys(), 2):
	for p in product(envs[a], envs[b]):
		sinter.append(tuple(np.sort(p)))

swithin = {s:list(combinations(envs[s], 2)) for s in envs.keys()}


figure()
nb_bins = 10
clrs = ['blue', 'red', 'green']
for i, e in enumerate(swithin.keys()):
	subplot(2,2,i+1)
	tmp = diffsess[swithin[e]].values.astype(np.float32).flatten()
	tmp2 = rnddiffsess[swithin[e]].values.astype(np.float32).flatten()
	title(e)
	hist(np.rad2deg(tmp), np.linspace(0, 180, nb_bins), density=True, color = clrs[0], alpha = 0.5, histtype = 'step')
	hist(np.rad2deg(tmp2), np.linspace(0, 180, nb_bins), density=True, color = 'grey', alpha = 0.5, histtype = 'step')
	xlim(0, 180)
subplot(224)
tmp = diffsess[sinter].values.astype(np.float32).flatten()
tmp2 = rnddiffsess[sinter].values.astype(np.float32).flatten()
title('Inter')
hist(np.rad2deg(tmp), np.linspace(0, 180, nb_bins), density=True, color = clrs[2], alpha = 0.5, histtype = 'step')
hist(np.rad2deg(tmp2), np.linspace(0, 180, nb_bins), density=True, color = 'grey', alpha = 0.5, histtype = 'step')
xlim(0, 180)



























#################################################################################################
# FITTING NORMAL DISTRIBUTION TO TC
#################################################################################################
from scipy import optimize
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

std = pd.DataFrame(index = list(alltc.keys()), columns = range(cellreg.shape[1]))

for n in alltc.keys():	
	tmp = alltc[n]
	ctc = centerTuningCurves(tmp.dropna(1))	
	for k in ctc.columns:
		try:
			popt, _ = optimize.curve_fit(gaussian, ctc.index.values, ctc[k].values)
		except RuntimeError:
			popt = np.ones(3)*np.nan
		std.loc[n,k] = popt[-1]

# plot(ctc[k])
# plot(ctc[k].index.values, gaussian(ctc[k].index.values, *popt))

# CUTTING IN 2 groups
std[std>0.4] = np.nan
std = std.dropna(how='all')
std = std[std.isna().sum(1)<4]
alltc2 = {}
for k in std.index.values:
	alltc2[k] = alltc[k]

alltc3 = {}
for k in list(set(alltc.keys()) - set(alltc2.keys())):
	alltc3[k] = alltc[k]

diffsess2 = computePairwiseAngularDifference(alltc2, sessions = np.arange(len(SF)))
diffsess3 = computePairwiseAngularDifference(alltc3, sessions = np.arange(len(SF)))

# RANDOMIZING
rnddiffsess2 = []
for k in range(200):
	print(k)
	rndcellreg = np.copy(cellreg[list(alltc2.keys())])
	for t in range(rndcellreg.shape[1]):		
		np.random.shuffle(rndcellreg[:,t])
	rndtc = {}
	for i in range(len(rndcellreg)):
		rndtc[i] = pd.DataFrame(columns = np.arange(rndcellreg.shape[1]))
		for j in np.where(rndcellreg[i]!=-1)[0]:
			rndtc[i][j] = TC[j][rndcellreg[i,j]]

	rnddiffsess2.append(computePairwiseAngularDifference(rndtc, sessions = np.arange(len(SF))))

rnddiffsess2 = pd.concat(rnddiffsess2)



figure()
gs = GridSpec(3,3)
nb_bins = 15
clrs = ['blue', 'red', 'green']
titles = ['Small', 'Large', 'Inter']
subplot(gs[0,0])
gr2 = []
for n in alltc2.keys():
	ctc = centerTuningCurves(alltc2[n].dropna(1))	
	plot(ctc, alpha = 0.5, color = 'grey')
	gr2.append(ctc)
gr2 = pd.concat(gr2, 1)
plot(gr2.mean(1))
subplot(gs[0,1])
gr3 = []
for n in alltc3.keys():
	ctc = centerTuningCurves(alltc3[n].dropna(1))	
	plot(ctc, alpha = 0.5, color = 'grey')
	gr3.append(ctc)
gr3 = pd.concat(gr3, 1)	
plot(gr3.mean(1), color = 'orange')
subplot(gs[0,2])
plot(gr2.mean(1))
plot(gr3.mean(1))


for i, ev in enumerate([scyl, slrg, sinter]):
	subplot(gs[1,i])
	tmp = diffsess2[ev].values.astype(np.float32).flatten()
	tmp2 = rnddiffsess2[ev].values.astype(np.float32).flatten()
	title(titles[i])
	hist(np.rad2deg(tmp), np.linspace(0, 180, nb_bins), density=True, color = clrs[i], alpha = 0.5, histtype = 'step')
	hist(np.rad2deg(tmp2), np.linspace(0, 180, nb_bins), density=True, color = 'grey', alpha = 0.5, histtype = 'step')
	xlim(0, 180)
subplot(gs[2,0])
for i, gr in enumerate([scyl, slrg, sinter]):
	tmp, bin_edges = np.histogram(diffsess2[gr].values.astype(np.float32).flatten(), np.linspace(0, np.pi, nb_bins), density=True)
	plot(bin_edges[0:-1], tmp*np.diff(bin_edges), color = clrs[i])












































group1 = tmp[tmp[(0,5)]<np.deg2rad(40)].index.values
group2 = tmp[tmp[(0,5)]>np.deg2rad(40)].index.values

figure()
gs = GridSpec(2, tmp.shape[1])
for j, gr in enumerate([group1, group2]):
	for i, k in enumerate(tmp.columns):
		subplot(gs[j,i])
		hist(np.rad2deg(tmp.loc[gr,[k]].values.astype(np.float32)))
		axvline(25)
		title(k)
		xlim(0, 180)

figure()
subplot(121)
hist(np.rad2deg(tmp.loc[group1].values.astype(np.float32).flatten()), 30)
xlim(0, 180)
subplot(122)
hist(np.rad2deg(tmp.loc[group2].values.astype(np.float32).flatten()), 30)
xlim(0, 180)




figure()
index = group1
for i, n in enumerate(index):
	subplot(int(np.ceil(np.sqrt(len(index)))),int(np.ceil(np.sqrt(len(index)))),i+1, projection='polar')	
	plot(alltc[n], color = 'red', alpha = 0.5)
	xticks([])
	yticks([])

figure()
index = group2
for i, n in enumerate(index):
	subplot(int(np.ceil(np.sqrt(len(index)))),int(np.ceil(np.sqrt(len(index)))),i+1, projection='polar')	
	plot(alltc[n], color = 'green', alpha = 0.5)
	xticks([])
	yticks([])




figure()
subplot(121)
tmp = np.rad2deg(diffpeaks.values.astype(np.float32).flatten())
hist(tmp, 30)
title('Diff same neurons peak')
subplot(122)
tmp = np.rad2deg(diffsess.values.astype(np.float32).flatten())
hist(tmp, 30)
title('Diff diff pairs')

figure()
gs = GridSpec(len(SF), len(SF))
for i, j in diffpeaks.columns:
	subplot(gs[i,j])
	hist(diffpeaks[(i,j)], 20)

figure()
gs = GridSpec(len(SF), len(SF))
for i, j in diffpeaks.columns:
	subplot(gs[i,j])
	hist(diffsess[(i,j)], 20)






#####################################################################################
# PLOT SPATIAL FOOTPRINTS WITH PREFERED DIRECTION
#####################################################################################
xcut = (40,260)
ycut = (20,230)

AllA = []

figure()
for i in range(len(SF)):
	subplot(3,4,i+1)
	peaks = PK[i][HD[i]]
	H2 = peaks.values/(2*np.pi)
	HSV2 = np.vstack((H2, np.ones_like(H2), np.ones_like(H2))).T
	RGB2 = hsv_to_rgb(HSV2)
	colorA = np.ones((dims[0], dims[1], 3))
	for j, n in enumerate(peaks.index.values):
		colorA[SF[i][n].T > 3] = RGB2[j]

	nonhd = list(set(np.arange(len(SF[i]))) - set(HD[i]))
	for j, n in enumerate(nonhd):
		colorA[SF[i][n].T > 4] = [0,0,0]

	imshow(colorA[ycut[0]:ycut[1],xcut[0]:xcut[1]])

	for n in peaks.index:
		_ , _, _ , maxloc = cv2.minMaxLoc(SF[i][n].T[ycut[0]:ycut[1],xcut[0]:xcut[1]])
		annotate(str(n), (maxloc[0], maxloc[1]), (maxloc[0]+1+np.random.rand(), maxloc[1]-1-np.random.rand()), fontsize = 1)
	
	AllA.append(colorA[ycut[0]:ycut[1],xcut[0]:xcut[1]])




sys.exit()

####################################################################################
# FIGURES
####################################################################################
figure()
hist(np.rad2deg(diffpeaks.median(1)),20)
axvline(20)

group1 = diffpeaks[diffpeaks.median(1)<np.deg2rad(20)].index.values
group2 = diffpeaks[diffpeaks.median(1)>np.deg2rad(20)].index.values


figure()
index = group1
for i, n in enumerate(index):
	subplot(int(np.ceil(np.sqrt(len(index)))),int(np.ceil(np.sqrt(len(index)))),i+1, projection='polar')	
	plot(alltc[n], color = 'red', alpha = 0.5)
	xticks([])
	yticks([])

figure()
index = group2
for i, n in enumerate(index):
	subplot(int(np.ceil(np.sqrt(len(index)))),int(np.ceil(np.sqrt(len(index)))),i+1, projection='polar')	
	plot(alltc[n], color = 'green', alpha = 0.5)
	xticks([])
	yticks([])

index = group1[np.argsort(n_sessions_detected[group1])[::-1]][0:10]
figure()
gs = GridSpec(len(index),n_sessions_detected.max())
for i,n in enumerate(index):
	for j in range(n_sessions_detected.max()):
		subplot(gs[i,j],projection = 'polar')
		plot(alltc[n][j], color = 'red')
		xticks([])
		yticks([])

index = group2[np.argsort(n_sessions_detected[group2])[::-1]][0:10]
figure()
gs = GridSpec(len(index),n_sessions_detected.max())
for i,n in enumerate(index):
	for j in range(n_sessions_detected.max()):
		subplot(gs[i,j],projection = 'polar')
		plot(alltc[n][j], color = 'green')
		xticks([])
		yticks([])





#####################################################################################
# PLOT CONCATENATED SPATIAL FOOTPRINT WITH CELL REG OUTPUT
#####################################################################################

AllA = np.concatenate(AllA, 1)
idx = np.argsort(n_sessions_detected)[::-1]

figure()
imshow(AllA)
for i in idx[0:3]:
	x = []
	y = []
	for j in range(len(cellreg[i])):
		if cellreg[i,j] > -1:
			tmp = SF[j][cellreg[i,j]].T[ycut[0]:ycut[1],xcut[0]:xcut[1]]
			_ , _, _ , maxloc = cv2.minMaxLoc(tmp)
			x.append(maxloc[0]+tmp.shape[1]*j)
			y.append(maxloc[1])
		plot(x, y, '.-', alpha = 0.5, linewidth = 1)
title("CellREG")



# figure()
# subplot(221)
# tmp = np.rad2deg(diffpeaks.loc[group1].values.astype(np.float32).flatten())
# hist(tmp, 30, color = 'red')
# title('Diff same neurons peak group1')
# subplot(222)
# tmp = np.rad2deg(diffpeaks.loc[group2].values.astype(np.float32).flatten())
# hist(tmp, 30, color = 'green')
# title('Diff same neurons peak group2')
# subplot(223)
# tmp = np.rad2deg(diffsess.loc[list(combinations(group1,2))].values.astype(np.float32).flatten())
# hist(tmp, 30, color = 'red')
# title('Diff diff pairs')
# subplot(224)
# tmp = np.rad2deg(diffsess.loc[list(combinations(group2,2))].values.astype(np.float32).flatten())
# hist(tmp, 30, color = 'green')
# title('Diff diff pairs')

#####################################################################################
# PLACE FIELDS
#####################################################################################
index = np.array(list(allpf.keys()))
index = index[np.argsort(n_sessions_detected[index])[::-1]]
for k in index[20:40]:
	figure(figsize = (30, 10))
	gs = GridSpec(2,n_sessions_detected.max())	
	for j in range(n_sessions_detected.max()):
		subplot(gs[1,j])		
		imshow(scipy.ndimage.gaussian_filter(allpf[k][j], (2, 2)))

		subplot(gs[0,j], projection = 'polar')		
		plot(alltc[k][j])
	show(block=True)


figure()
for i in range(130):
	subplot(10, 30, i+1, projection = 'polar')
	plot(TC[13][i])