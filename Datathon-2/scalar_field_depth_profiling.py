import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as manimation
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Indian Ocean - Temperature', artist='Swasti',
                comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)

# List of files to be visualised
current_data_files = glob.glob("data/datathon2_data/OneDrive_1_12-09-2020/PotentialTemperature_3D/*.txt")[:1]
current_file = current_data_files[0]

BAD_FLAG = '-1.E+34'

idx = 0

LAT = set()
LON = set()
DEP = set()
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
                try:
                    date = data[0]
                    lon = float(data[2])
                    lat = float(data[3])
                    dep = float(data[4])

                    # If the data is a BAD_FLAG, convert it into NaN (so that it is ignored by matplotlib)
                    temp = np.nan

                    if data[5] != BAD_FLAG:
                        temp = float(data[5])

                    OCEAN[lon,lat,dep] = temp
                    LAT.add(lat)
                    LON.add(lon)
                    DEP.add(dep)
                except:
                    continue

        else:
            break


        idx += 1


LON = list(LON)
LAT = list(LAT)
DEP = list(DEP)

LON.sort()
LAT.sort()
DEP.sort()


def update(lon):
    
    # Convert TEMP into grid format
    TEMP = np.zeros((len(LAT),len(DEP)),np.float)

    for i in range(len(LAT)):
        for j in range(len(DEP)):
            try:
                TEMP[i][j] = OCEAN[lon,LAT[i],DEP[j]]
            except:
                TEMP[i][j] = np.nan
                continue

    # Visualize the data
    plt.clf()
    lat, dep = np.meshgrid(LAT, DEP)

    h = plt.contourf(lat,dep,TEMP.T,cmap=cm.hot)
    cbar = plt.colorbar()
    cbar.set_label("Potential Temperature (degree Celcius)")
    plt.title("Indian Ocean Potential Temperature at longitude = {}mdegrees on {}".format(lon,date.strip("\"")))
    plt.xlabel("Latitude in degrees")
    plt.ylabel("Depth in meters")

    return h


fig = plt.figure(figsize=(16,8))

with writer.saving(fig, "writer_test.mp4", dpi=250):
    for lon in LON:
        update(lon)
        writer.grab_frame()
