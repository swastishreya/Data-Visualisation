import glob
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# List of files to be visualised
ssha_data_files = glob.glob("data/ssha/*.txt")
ssha_data_files.sort()

BAD_FLAG = '-1.E+34'

idx = 0

# Data structure to store the value of SSHA at location (LON,LAT)
OCEAN = dict()
date = ""

with open(ssha_data_files[0],'r') as f:
    while(f):
        r = f.readline()
        if r != '':
            if idx >= 10:
                data = r.strip().split(',')
                date = data[0]
                lon = float(data[2])
                lat = float(data[3])

                # If the data is a BAD_FLAG, convert it into NaN (so that it is ignored by matplotlib)
                ssha = np.nan
                if data[4] != BAD_FLAG:
                    ssha = float(data[4])
                if lon not in OCEAN:
                    OCEAN[lon] = dict()
                OCEAN[lon][lat] = ssha

        else:
            break


        idx += 1

LAT = []
LON = []

for lon in OCEAN:
    LON.append(lon)
    for lat in OCEAN[lon]:
        LAT.append(lat)

LON = list(set(LON))
LAT = list(set(LAT))

LON.sort()
LAT.sort()

# Convert SSHA into grid format
SSHA = np.zeros((len(LON),len(LAT)),np.float)

for i in range(len(LON)):
    for j in range(len(LAT)):
        SSHA[i][j] = OCEAN[LON[i]][LAT[j]]

# Visualize the data
fig = plt.figure()
ax = fig.gca(projection='3d')

# map = Basemap(projection='cyl',llcrnrlon=min(LON),llcrnrlat=min(LAT),urcrnrlon=max(LON),urcrnrlat=max(LAT),lat_0=0,lon_0=74.9544)
lon, lat = np.meshgrid(LON, LAT)
# map.drawcoastlines()
# map.drawparallels(np.arange(-90., 90., 10.), linewidth=2, labels=[1,0,0,0])
# map.drawmeridians(np.arange(-180., 180., 10.), linewidth=2, labels=[0,0,0,1])

h = ax.plot_surface(lon,lat,SSHA.T,cmap=cm.hot)
# plt.colorbar()
ax.set_title("Indian Ocean SSHA on {}".format(date.strip("\"")))

plt.show()

