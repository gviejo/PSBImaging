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
from wrappers import *
from functions import *
from scipy.ndimage.filters import gaussian_filter
from matplotlib.gridspec import GridSpecFromSubplotSpec

def butter_bandpass(lowcut, highcut, fs, order=5):
	from scipy.signal import butter
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	from scipy.signal import lfilter
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y




path = '/mnt/Data2/A9800/A9802/A9802-210910'

name = path.split('/')[-1]
dims = (181,132)
files = os.listdir(path)
fs = 30000
n_channels = 43
# LOAD BINARY FILE
bfile = name+'.dat'
f = open(os.path.join(path, bfile), 'rb')
startoffile = f.seek(0, 0)
endoffile = f.seek(0, 2)
bytes_size = 2        
n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
f.close()
fp = np.memmap(os.path.join(path, bfile), np.int16, 'r', shape = (n_samples, n_channels))		
timestep = np.arange(0, n_samples)/fs

# loading LFP
lfp = nts.Tsd(t = timestep, d = fp[:,22], time_units = 's')

# loading acceleromter
accel = nts.Tsd(t = timestep, d = fp[:,32], time_units = 's')

# opto ttl
opto = fp[:,35][:]
opto = np.array(opto).astype(np.int32)
opto = opto-opto.min()
start,_ = scipy.signal.find_peaks(np.diff(opto), height=15000)
end,_ = scipy.signal.find_peaks(np.diff(opto)*-1, height=15000)
start -= 1
opto_ep = nts.IntervalSet(start = timestep[start], end = timestep[end], time_units = 's')
opto_ep = opto_ep.merge_close_intervals(30000)
opto = nts.Tsd(t = timestep, d = opto, time_units = 's')

# Calcium ttl
calc = fp[:,36][:]
calc = np.array(calc).astype(np.int32)
calc = calc - calc.min()
calc = np.abs(np.diff(calc))
peaks,_ = scipy.signal.find_peaks(calc, height=30000, distance = 1000)
ttl_miniscope = pd.Series(index = timestep[peaks], data = calc[peaks])


# loading calcium transients
C_file = [f for f in files if 'C.csv' in f][0]
C = pd.read_csv(os.path.join(path, C_file), header = None)
C = C.T
C = pd.DataFrame(index = C.index, columns = np.arange(C.shape[1]-1), data = C.values[:,1:])

C.columns = pd.Index(np.arange(C.shape[1]))
C = C.iloc[0:np.minimum(len(ttl_miniscope),len(C))]
C.index = nts.Ts(t = ttl_miniscope.index.values[0:np.minimum(len(ttl_miniscope),len(C))], time_units = 's').index

C = nts.TsdFrame(C)

# loading spatial footprints
A_file = [f for f in files if '_A.csv' in f][0]
A = pd.read_csv(os.path.join(path, A_file), header = None)		
A = A.values
A = A.T.reshape(A.shape[1], dims[0], dims[1])
A = A[1:,:,:]

# filter for ripples
frequency = fs
low_cut = 100
high_cut = 300

signal = butter_bandpass_filter(lfp.values, low_cut, high_cut, frequency, order = 4)

flfp = nts.Tsd(t = lfp.index.values, d = signal)

cn = pd.read_csv('/mnt/Data2/A9800/A9802/A9802-210910/A9802-210910-cn.csv').T


rip_ex = 14
ep2 = nts.IntervalSet(start = [opto_ep.loc[rip_ex,'start']-2e5], end = [opto_ep.loc[rip_ex,'start']+2e5])
figure(figsize = (15,5))
cmap = plt.get_cmap("tab10")
outer = GridSpec(1,2,width_ratios = [0.4,0.6], wspace = 0.4)
ax = subplot(outer[0,0])
# colorA = np.zeros((dims[0], dims[1], 3))
# colorA *= np.nan
# for i in range(len(A)):
# 	colorA[A[i] > 4] = list(cmap(i))[0:3]

imshow(cn.values[0:125,0:120])
max_pt = np.array([cv2.minMaxLoc(A[i])[3] for i in range(len(A))])
clrs = [cmap(i) for i in range(len(A))]
scatter(max_pt[:,0], max_pt[:,1], c = clrs, s= 80)
title("Local Correlation")
# for i, txt in enumerate(np.arange(C.shape[1])):
#     gca().annotate(str(txt), (max_pt[i,0], max_pt[i,1]), fontsize = 20)


#imshow(colorA[0:125,0:120])
ax = subplot(outer[0,1])
gs = GridSpecFromSubplotSpec(3,1,ax)
# zoom lfp
subplot(gs[0,0])
simpleaxis(gca())
plot(flfp.restrict(ep2))
ylabel("CA1 LFP\n(100-300 Hz)", rotation = 0, labelpad=20)
xt = ep2.iloc[0,0] + np.arange(0, ep2.tot_length('s'), 0.1)*1000*1000
xtl = np.round(np.arange(0, ep2.tot_length('s'), 0.1), 1)
xticks(xt, xtl)
axvspan(opto_ep.loc[rip_ex,'start'], ep2.loc[0,'end'], alpha = 0.24, color = 'green')

# filtered lfp
ep = nts.IntervalSet(start = [opto_ep.loc[rip_ex,'start']-1e6], end = [opto_ep.loc[rip_ex+1,'end']+6e5])
xt = ep.iloc[0,0] + np.arange(0, ep.tot_length('s'), 1)*1000*1000
xtl = np.arange(0, ep.tot_length('s'), 1, dtype = np.int) 

subplot(gs[1,0])
simpleaxis(gca())
plot(flfp.restrict(ep))
ylabel("CA1 LFP\n(100-300 Hz)", rotation = 0, labelpad=20)
xticks([], [])
axvspan(opto_ep.loc[rip_ex,'start'], opto_ep.loc[rip_ex,'end'], alpha = 0.24, color = 'green')
axvspan(opto_ep.loc[rip_ex+1,'start'], opto_ep.loc[rip_ex+1,'end'], alpha = 0.24, color = 'green')

# calcium traces
subplot(gs[2,0])
simpleaxis(gca())
for i in range(C.shape[1]):
	tmp = C[i].restrict(ep)	
	plot(tmp.index.values, (tmp.values*3)+i, color = cmap(i))
ylabel("Calcium \n Transients", rotation = 0, labelpad=60)
xticks(xt, xtl)
xlabel("Time (s)")
axvspan(opto_ep.loc[rip_ex,'start'], opto_ep.loc[rip_ex,'end'], alpha = 0.24, color = 'green')
axvspan(opto_ep.loc[rip_ex+1,'start'], opto_ep.loc[rip_ex+1,'end'], alpha = 0.24, color = 'green')
savefig('../figures/figure_supermouse.pdf', format = 'pdf', dpi = 200, bbox_inches = 'tight')



