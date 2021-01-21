"""
===========
MovieWriter
===========

This example uses a MovieWriter directly to grab individual frames and write
them to a file. This avoids any event loop integration, but has the advantage
of working with even the Agg backend. This is not recommended for use in an
interactive setting.

"""
# -*- noplot -*-

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import h5py as hd


FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=30, metadata=metadata)


# plt.xlim(-5, 5)
# plt.ylim(-5, 5)

# x0, y0 = 0, 0

folder_name = '/mnt/LocalHDD/MINISCOPE/A0633/2020_11_13/18_45_42/Miniscope'
data 		= hd.File(folder_name+"/motion_corrected.hdf5", 'r')

dims = (608,608)
fig = plt.figure()
im = plt.imshow(data['movie'][0].reshape(dims))

# import sys
# sys.exit()



with writer.saving(fig, "writer_test.mp4", 2000):
    for i in range(2000):
        print(i)
        im.set_data(data['movie'][i].reshape(dims))
        writer.grab_frame()
