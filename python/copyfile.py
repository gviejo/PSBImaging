import numpy as np
import sys, os
from shutil import copy2



# path = '/mnt/Hypernova/Sofia/A6500/A6512'
# target = '/media/guillaume/Elements/A6500/A6512'
path = '/mnt/Hypernova/Guillaume/A0600/A0643'
target = '/media/guillaume/A0600/A0643'

lf = os.listdir(path)

for l in lf:
	if l.split('-')[1] >= '210823':
		#print(l)
		p = os.path.join(path, l)
		if os.path.isdir(p):
			pt = os.path.join(target, l)
			if not os.path.exists(pt):
				os.mkdir(pt)

				toc = ['_0.csv', '_0_analogin.dat', '_raw.avi']
				for t in toc:
					po = os.path.join(p, l+t)
					print(po)
					if os.path.exists(po):
						copy2(po, pt)