import glob
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Indian Ocean - SSHA', artist='Swasti',
                comment='Movie support!')
writer = FFMpegWriter(fps=12, metadata=metadata)

# List of files to be visualised
data_files = glob.glob("data/ssha/*.txt")
data_files.sort()

BAD_FLAG = '-1.E+34'

def update(data_file):
    idx = 0

    # Data structure to store the value of SSHA at location (LON,LAT)
    OCEAN = dict()
    date = ""

    with open(data_file,'r') as f:
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
    plt.clf()
    map = Basemap(projection='cyl',llcrnrlon=min(LON),llcrnrlat=min(LAT),urcrnrlon=max(LON),urcrnrlat=max(LAT),lat_0=0,lon_0=74.9544)
    lon, lat = np.meshgrid(LON, LAT)
    map.drawcoastlines()
    map.drawparallels(np.arange(-90., 90., 10.), linewidth=2, labels=[1,0,0,0])
    map.drawmeridians(np.arange(-180., 180., 10.), linewidth=2, labels=[0,0,0,1])

    h = map.contourf(lon,lat,SSHA.T,levels=np.linspace(-0.44,0.44,100),cmap=cm.BrBG)
    cbar = plt.colorbar()
    cbar.set_label("Relative hight of sea suface")
    plt.title("Indian Ocean Sea Surface Height Anomaly on {}".format(date.strip("\"")))

    return h


fig = plt.figure(figsize=(16,8))

with writer.saving(fig, "writer_test.mp4", dpi=100):
    for f in data_files:
        update(f)
        writer.grab_frame()
