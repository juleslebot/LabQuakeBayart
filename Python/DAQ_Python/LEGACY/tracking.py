import trackpy as tp


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims

mpl.rc('image', cmap='gray')

## Open images
location = "D:/Users/Manips/Documents/DATA/FRICS/2021-02-19/smol_cine.tif"

frames=pims.open(location)


# to display a frame
#plt.imshow(frames[0])

frames[0].metadata['DateTime']

##

f = tp.locate(frames[0], 21, invert=False)

tp.annotate(f, frames[0])

##



fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)

# Optionally, label the axes.
ax.set(xlabel='mass', ylabel='count');
plt.show()
