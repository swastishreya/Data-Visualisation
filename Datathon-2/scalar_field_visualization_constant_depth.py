import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as manimation
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Indian Ocean - Salinity', artist='Swasti',
                comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)

# List of files to be visualised
current_data_files = glob.glob("data/datathon2_data/OneDrive_1_12-09-2020/Salinity_3D/*.txt")[:20]
current_data_files.sort()

BAD_FLAG = '-1.E+34'

def update(current_file):
    idx = 0

    LAT = set()
    LON = set()
    # Data structure to store the value of current at location (LON,LAT)
    OCEAN = dict()
    date = ""

    idx = 0
    with open(current_file,'r') as f:
        while(f):
            r = f.readline()
            if r != '':
                if idx >= 11:
                    data = r.strip().split(',')
                    date = data[0]
                    lon = float(data[2])
                    lat = float(data[3])
                    dep = float(data[4])

                    if dep != 45.0:
                        continue
                    # If the data is a BAD_FLAG, convert it into NaN (so that it is ignored by matplotlib)
                    salt = np.nan

                    if data[5] != BAD_FLAG:
                        salt = float(data[5])

                    OCEAN[lon,lat] = salt
                    LAT.add(lat)
                    LON.add(lon)

            else:
                break


            idx += 1


    LON = list(LON)
    LAT = list(LAT)

    LON.sort()
    LAT.sort()

    # Convert SALT into grid format
    SALT = np.zeros((len(LON),len(LAT)),np.float)

    for i in range(len(LON)):
        for j in range(len(LAT)):
            try:
                SALT[i][j] = OCEAN[LON[i],LAT[j]]
            except:
                SALT[i][j] = np.nan
                continue

    # Visualize the data
    plt.clf()
    map = Basemap(projection='cyl',llcrnrlon=min(LON),llcrnrlat=min(LAT),urcrnrlon=max(LON),urcrnrlat=max(LAT),lat_0=0,lon_0=74.9544)
    lon, lat = np.meshgrid(LON, LAT)
    map.drawcoastlines()
    map.drawparallels(np.arange(-90., 90., 10.), linewidth=2, labels=[1,0,0,0])
    map.drawmeridians(np.arange(-180., 180., 10.), linewidth=2, labels=[0,0,0,1])

    h = map.contourf(lon,lat,SALT.T,cmap=cm.plasma)
    cbar = plt.colorbar()
    cbar.set_label("Salinity (psu)")
    plt.title("Indian Ocean Salinity at depth = 45.0m on {}".format(date.strip("\"")))

    return h


fig = plt.figure(figsize=(16,8))

with writer.saving(fig, "writer_test.mp4", dpi=250):
    for f in current_data_files:
        update(f)
        writer.grab_frame()
