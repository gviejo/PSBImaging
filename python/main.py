import numpy as np
from time import time
import scipy
import glob
import yaml
import sys,os
import h5py as hd
from time import time
import av

from miniscopy.base.motion_correction import *
from miniscopy import setup_cluster, CNMFE

folder_name = '/mnt/LocalHDD/MINISCOPE/A0633/2020_11_13/18_45_42/Miniscope'
files = glob.glob(folder_name+'/*.avi')

#############################################################################################################
# LOADING PARAMETERS
#############################################################################################################
parameters = yaml.load(open(folder_name+'/parameters.yaml', 'r'))

#############################################################################################################
# start a cluster for parallel processing
#############################################################################################################
c, procs, n_processes = setup_cluster(backend='local', n_processes=8, single_thread=False)
sys.exit()
# #############################################################################################################
# # MOTION CORRECTION | create the motion_corrected.hdf5 file
# #############################################################################################################
data, video_info = normcorre(files, procs, parameters['motion_correction'])

data = hd.File(folder_name+"/motion_corrected.hdf5", 'r')

#############################################################################################################
# CONSTRAINED NON NEGATIVE MATRIX FACTORIZATION
#############################################################################################################
# parameters['cnmfe']['init_params']['thresh_init'] = 1.2
# parameters['cnmfe']['init_params']['min_corr'] = 0.8
# parameters['cnmfe']['init_params']['min_pnr'] = 1.5

cnm = CNMFE(data, parameters['cnmfe'])

cnm.fit(procs)
#############################################################################################################
# VISUALIZATION
#########################################################################################################
cn, pnr = cnm.get_correlation_info()

dims = cnm.dims
C = cnm.C.value.copy()
A = cnm.A.value.copy()

# A is normalized to 1 for display
A -= np.vstack(A.min(1))
A /= np.vstack(A.max(1))
Atotal = A.sum(0).reshape(dims)


from matplotlib.pyplot import *
import matplotlib.gridspec as gridspec

tmp = Atotal.copy()
tmp[tmp == 0] = np.nan
figure(figsize = (15,5))
gs = gridspec.GridSpec(3,3)
subplot(gs[0:2,0])
imshow(Atotal)
subplot(gs[0:2,1])
imshow(cn)
contour(np.flip(tmp, 0), origin = 'upper', cmap = 'gist_gray')
subplot(gs[0:2,2])
imshow(cn)
contour(np.flip(tmp, 0), origin = 'upper', cmap = 'gist_gray')
subplot(gs[-1,:])
plot(C)

show()
