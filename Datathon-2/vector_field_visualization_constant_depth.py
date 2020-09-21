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

# List of files to be visualised
meridional_current_data_files = glob.glob("data/datathon2_data/OneDrive_1_12-09-2020/meridional-current_3D/*.txt")[:20]
meridional_current_data_files.sort()

BAD_FLAG = '-1.E+34'

def update(current_file):
    idx = 0

    # Data structure to store the value of current at location (LON,LAT)
    OCEAN = dict()
    date = ""

    meridional_current_file = "data/datathon2_data/OneDrive_1_12-09-2020/meridional-current_3D/"+current_file
    zonal_current_file = "data/datathon2_data/OneDrive_1_12-09-2020/zonal-current_3D/"+current_file

    with open(meridional_current_file,'r') as f:
        while(f):
            r = f.readline()
            if r != '':
                if idx >= 12:
                    data = r.strip().split(',')
                    date = data[0]
                    lon = float(data[2])
                    lat = float(data[3])
                    dep = float(data[4])

                    if dep != 5.0:
                        continue
                    # If the data is a BAD_FLAG, convert it into NaN (so that it is ignored by matplotlib)
                    meridional_current = np.nan

                    if data[5] != BAD_FLAG:
                        meridional_current = float(data[5])*-1
                    if lon not in OCEAN:
                        OCEAN[lon] = dict()
                    OCEAN[lon][lat] = [0,meridional_current]

            else:
                break


            idx += 1

    idx = 0
    with open(zonal_current_file,'r') as f:
        while(f):
            r = f.readline()
            if r != '':
                if idx >= 12:
                    data = r.strip().split(',')
                    lon = float(data[2])
                    lat = float(data[3])
                    dep = float(data[4])

                    if dep != 5.0:
                        continue
                    # If the data is a BAD_FLAG, convert it into NaN (so that it is ignored by matplotlib)
                    zonal_current = np.nan

                    if data[5] != BAD_FLAG:
                        zonal_current = float(data[5])*-1

                    OCEAN[lon][lat][0] = zonal_current

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

    LON1 = LON#[::2]
    LAT1 = LAT#[::2]

    # Convert meridional_current into grid format
    meridional_current = np.zeros((len(LON1),len(LAT1)),np.float)
    zonal_current = np.zeros((len(LON1),len(LAT1)),np.float)

    for i in range(len(LON1)):
        for j in range(len(LAT1)):
            try:
                zc = OCEAN[LON1[i]][LAT1[j]][0]
                mc = OCEAN[LON1[i]][LAT1[j]][1]
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
    lon1, lat1 = np.meshgrid(LON1, LAT1)
    map.drawcoastlines()
    map.drawparallels(np.arange(-90., 90., 10.), linewidth=2, labels=[1,0,0,0])
    map.drawmeridians(np.arange(-180., 180., 10.), linewidth=2, labels=[0,0,0,1])
    q = map.quiver(lon1,lat1,zonal_current.T,meridional_current.T,width=0.001, color='black', scale=150)    
    _ = plt.quiverkey(q, 0.85, 0.85, 2,'2 m/sec', labelpos='E',coordinates='figure')
    plt.title("Currents (zonal and meridional) at depth = 5.0m in Indian Ocean on {}".format(date.strip("\"")))

    return q


fig = plt.figure(figsize=(16,8))

with writer.saving(fig, "writer_test.mp4", dpi=250):
    for f in meridional_current_data_files:
        f = f[64:]
        update(f)
        writer.grab_frame()
