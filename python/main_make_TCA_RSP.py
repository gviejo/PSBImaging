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
# from pycircstat.descriptive import mean as circmean
from matplotlib.colors import hsv_to_rgb
#import cv2
from matplotlib.gridspec import GridSpec
from itertools import product, combinations
from scipy.stats import norm
from scipy.ndimage.filters import gaussian_filter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import tensortools as tt
# import tensorly as tl
# tl.set_backend('pytorch')


data_directory = '/mnt/DataRAID/MINISCOPE'
#data_directory = '/media/guillaume/Elements'

############################################################
# ANIMAL INFO
############################################################
fbasename = 'A6510'
info = pd.read_csv('/home/guillaume/PSBImaging/python/datasets_'+fbasename+'.csv', comment = '#', header = 5, delimiter = ',', index_col=False, usecols = [0,2,3,4]).dropna()
paths = [os.path.join(data_directory, fbasename[0:3] + '00', fbasename, fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '')) for i in info.index]
sessions = [fbasename+'-'+info.loc[i,'Recording day'][2:].replace('/', '') for i in info.index]
info['paths'] = paths
info['sessions'] = sessions
info = info.set_index('sessions')

if fbasename == 'A6509':
	dims = (202,192)
elif fbasename == 'A6510':
	dims = (192,251)



############################################################
# LOADING DATA
############################################################
SF, TC, PF, allinfo, positions, DFFS, Cs = loadDatas(paths, dims)


cellreg, scores = loadCellReg(os.path.join(data_directory, fbasename[0:3] + '00', fbasename))


############################################################
# SELECTING DATA
############################################################
n_sessions_detected = np.sum(cellreg!=-1, 1)
# sys.exit()
# Detected in at least 1 session
#tokeep = np.where(n_sessions_detected > 28)[0]
tokeep = np.where(n_sessions_detected == len(sessions))[0]
# #tokeep = np.where(n_sessions_detected == 20)[0]

# Making full data

decim = 3
nt = np.min([Cs[i].shape[0] for i in Cs.keys()]) 
ntm = int(np.ceil(nt/decim))
# MAKING THE DATA
data = np.zeros((ntm,len(tokeep),len(sessions)))
mask = np.zeros_like(data)
for i in Cs.keys():
	for j, n in enumerate(tokeep):
		if cellreg[n,i] != -1:
			tmp = Cs[i][cellreg[n,i]].values[0:nt]
			tmp = np.sqrt(tmp)
			# tmp = tmp - tmp.mean()
			# tmp = tmp / tmp.std()
			tmp = scipy.ndimage.gaussian_filter1d(tmp, 80)			
			data[:,j,i] = scipy.signal.decimate(tmp, decim)
			mask[:,j,i] = 1

#sys.exit()

# #########################################################
# # TENSORLY
# #########################################################
# from tensorly.decomposition import parafac
# num_components = 12

# tensor = tl.tensor(data, device = 'cuda', dtype = tl.float32)
# mask2 = tl.tensor(mask, device='cuda', dtype = tl.int32)
# fac = parafac(tensor, rank=num_components, mask = mask2, n_iter_max=20000, init = 'random', tol=1.0e-30, linesearch=True, verbose = 1, random_state = np.random.RandomState())
# #fac = parafac(data, rank=num_components, mask = mask, n_iter_max=20000, tol=1.0e-15, linesearch=True, verbose = 1)
# recdata = tl.cp_to_tensor(fac).to('cpu')

# figure()
# B, W, A = fac[1]
# B = np.array(B.to('cpu'))
# W = np.array(W.to('cpu'))
# A = np.array(A.to('cpu'))

# clrs = [list(np.unique(info.loc[sessions,'Rig'])).index(s) for s in info.loc[sessions,'Rig'].values]

# subplot(131)
# plot(B)
# title("Temporal factors")

# subplot(132)
# scatter(W[:,0], W[:,1])

# title("NEuron factors")

# subplot(133)
# scatter(A[:,0], A[:,1], c = clrs, cmap = 'jet')
# title("Session factors")

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap

# figure()
# subplot(221)
# tmp = PCA(n_components = 2).fit_transform(W)
# scatter(tmp[:,0], tmp[:,1])
# subplot(222)
# tmp = TSNE(n_components = 2, perplexity = 5).fit_transform(W)
# scatter(tmp[:,0], tmp[:,1])

# subplot(223)
# tmp = PCA(n_components = 2).fit_transform(A)
# scatter(tmp[:,0], tmp[:,1], c = clrs, cmap = 'jet')
# subplot(224)
# tmp = TSNE(n_components = 2, perplexity = 15).fit_transform(A)
# scatter(tmp[:,0], tmp[:,1], c = clrs, cmap = 'jet')

# show()



# cc = []
# for i in range(len(sessions)):
# 	a = np.corrcoef(data[:,:,i].T)
# 	cc.append(a[np.triu_indices_from(a)])
# cc = np.array(cc).T

# # tmp = TSNE(n_components = 2, perplexity = 10).fit_transform(cc.T)

# # tmp = PCA(n_components = 2).fit_transform(cc.T)

# # scatter(tmp[:,0], tmp[:,1], c = clrs)


# sys.exit()

#########################################################
# TENSORSTOOLS
#########################################################
num_components = 10

ensemble = tt.Ensemble(fit_method="ncp_hals", fit_options = {'tol':1e-20, 'max_iter':100000})
#ensemble.fit(data, ranks=range(1, num_components+1), replicates=4)
ensemble.fit(data, ranks=[num_components], replicates=4, verbose = True)

fig, axes = plt.subplots(1, 2)
tt.plot_objective(ensemble, ax=axes[0])   # plot reconstruction error as a function of num components.
tt.plot_similarity(ensemble, ax=axes[1])  # plot model similarity as a function of num components.
fig.tight_layout()

# Plot the low-d factors for an example model, e.g. rank-2, first optimization run / replicate.

replicate = 0
tt.plot_factors(ensemble.factors(num_components)[replicate])  # plot the low-d factors

plt.show()


figure()
B, W, A = ensemble.factors(num_components)[0]

clrs = [list(np.unique(info.loc[sessions,'Rig'])).index(s) for s in info.loc[sessions,'Rig'].values]

subplot(131)
plot(B)
title("Temporal factors")

subplot(132)
plot(range(W.shape[0]), W, 'o')

title("NEuron factors")

subplot(133)
scatter(A[:,0], A[:,1], c = clrs)
title("Session factors")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

figure()
subplot(121)
tmp = PCA(n_components = 2).fit_transform(A)
scatter(tmp[:,0], tmp[:,1], c = clrs, cmap = 'jet')
subplot(122)
tmp = TSNE(n_components = 2, perplexity = 5).fit_transform(A)
scatter(tmp[:,0], tmp[:,1], c = clrs, cmap = 'jet')

