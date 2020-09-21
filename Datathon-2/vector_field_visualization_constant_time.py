import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as manimation
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Indian Ocean - Currents', artist='Swasti',
                comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)

BAD_FLAG = '-1.E+34'

idx = 0

# Data structure to store the value of current at location (LON,LAT)
OCEAN = dict()
date = ""
LAT = set()
LON = set()
DEP = set()

meridional_curr_file_path = "data/datathon2_data/OneDrive_1_12-09-2020/meridional-current_3D/063_04_Nov_2004.txt"
zonal_curr_file_path = "data/datathon2_data/OneDrive_1_12-09-2020/zonal-current_3D/063_04_Nov_2004.txt"

with open(meridional_curr_file_path,'r') as f:
    while(f):
        r = f.readline()
        if r != '':
            if idx >= 12:
                data = r.strip().split(',')
                date = data[0]
                lon = float(data[2])
                lat = float(data[3])
                dep = float(data[4])

                # If the data is a BAD_FLAG, convert it into NaN (so that it is ignored by matplotlib)
                meridional_current = np.nan

                if data[5] != BAD_FLAG:
                    meridional_current = float(data[5])*-1
                if lon not in OCEAN:
                    OCEAN[lon] = dict()
                OCEAN[lon,lat,dep] = [0,meridional_current]

        else:
            break


        idx += 1

idx = 0
with open(zonal_curr_file_path,'r') as f:
    while(f):
        r = f.readline()
        if r != '':
            if idx >= 12:
                data = r.strip().split(',')
                lon = float(data[2])
                lat = float(data[3])
                dep = float(data[4])

                # If the data is a BAD_FLAG, convert it into NaN (so that it is ignored by matplotlib)
                zonal_current = np.nan

                if data[5] != BAD_FLAG:
                    zonal_current = float(data[5])*-1

                OCEAN[lon,lat,dep][0] = zonal_current
                LAT.add(lat)
                LON.add(lon)
                DEP.add(dep)

        else:
            break


        idx += 1


LON = list(LON)
LON.sort()
LAT = list(LAT)
LAT.sort()
DEP = list(DEP)
DEP.sort()


def update(dep):

    # Convert meridional_current into grid format
    meridional_current = np.zeros((len(LON),len(LAT)),np.float)
    zonal_current = np.zeros((len(LON),len(LAT)),np.float)

    for i in range(len(LON)):
        for j in range(len(LAT)):
            try:
                zc = OCEAN[LON[i],LAT[j],dep][0]
                mc = OCEAN[LON[i],LAT[j],dep][1]
                zonal_current[i][j] = zc
                meridional_current[i][j] = mc
            except:
                zonal_current[i][j] = np.nan
                meridional_current[i][j] = np.nan
                continue

    # Visualize the data
    plt.clf()
    map = Basemap(projection='cyl',llcrnrlon=min(LON),llcrnrlat=min(LAT),urcrnrlon=max(LON),urcrnrlat=max(LAT),lat_0=0,lon_0=74.9544)
    lon, lat = np.meshgrid(LON, LAT)
    map.drawcoastlines()
    map.drawparallels(np.arange(-90., 90., 10.), linewidth=2, labels=[1,0,0,0])
    map.drawmeridians(np.arange(-180., 180., 10.), linewidth=2, labels=[0,0,0,1])
    q = map.quiver(lon,lat,zonal_current.T,meridional_current.T,width=0.001, color='black', scale=150)    
    _ = plt.quiverkey(q, 0.85, 0.85, 2,'2 m/sec', labelpos='E',coordinates='figure')
    plt.title("Currents (zonal and meridional) at depth = {}m in Indian Ocean on {}".format(dep,date.strip("\"")))

    return q


fig = plt.figure(figsize=(16,8))

with writer.saving(fig, "writer_test.mp4", dpi=250):
    for dep in DEP:
        update(dep)
        writer.grab_frame()
