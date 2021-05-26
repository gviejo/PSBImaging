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
from matplotlib.gridspec import GridSpec
import h5py


# data = pd.HDFStore('/mnt/DataRAID/MINISCOPE/A6500/A6504/A6504-210412/A6504-210412.h5')

# data = pd.read_hdf('/mnt/DataRAID/MINISCOPE/A6500/A6504/A6504-210412/A6504-210412.h5')

f = h5py.File('/mnt/DataAdrienBig/PeyracheLabData/Sofia/A6500/A6504/A6504-210413/A6504-210413.h5', 'r')

a = f['mov'][0:4000,:,:]