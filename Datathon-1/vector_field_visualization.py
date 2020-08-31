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
meridional_current_data_files = glob.glob("data/meridional_current/*.txt")[:20]
meridional_current_data_files.sort()

BAD_FLAG = '-1.E+34'

def update(current_file):
    idx = 0

    # Data structure to store the value of current at location (LON,LAT)
    OCEAN = dict()
    date = ""

    meridional_current_file = "data/meridional_current/"+current_file
    zonal_current_file = "data/zonal_current/"+current_file

    with open(meridional_current_file,'r') as f:
        while(f):
            r = f.readline()
            if r != '':
                if idx >= 11:
                    data = r.strip().split(',')
                    date = data[0]
                    lon = float(data[2])
                    lat = float(data[3])

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
                if idx >= 11:
                    data = r.strip().split(',')
                    lon = float(data[2])
                    lat = float(data[3])

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

    LON1 = LON[::2]
    LAT1 = LAT[::2]

    # Convert meridional_current into grid format
    meridional_current = np.zeros((len(LON1),len(LAT1)),np.float)
    zonal_current = np.zeros((len(LON1),len(LAT1)),np.float)
    magnitude = np.zeros((len(LON),len(LAT)),np.float)

    for i in range(len(LON)):
        for j in range(len(LAT)):
            zc = OCEAN[LON[i]][LAT[j]][0]
            mc = OCEAN[LON[i]][LAT[j]][1]
            mag = np.sqrt(zc**2 + mc**2)
            magnitude[i][j] = mag

    for i in range(len(LON1)):
        for j in range(len(LAT1)):
            zc = OCEAN[LON1[i]][LAT1[j]][0]
            mc = OCEAN[LON1[i]][LAT1[j]][1]
            # mag = np.sqrt(zc**2 + mc**2)
            zonal_current[i][j] = (zc)#/mag)
            meridional_current[i][j] = (mc)#/mag)

    # Visualize the data
    plt.clf()
    map = Basemap(projection='cyl',llcrnrlon=min(LON),llcrnrlat=min(LAT),urcrnrlon=max(LON),urcrnrlat=max(LAT),lat_0=0,lon_0=74.9544)
    lon, lat = np.meshgrid(LON, LAT)
    lon1, lat1 = np.meshgrid(LON1, LAT1)
    map.drawcoastlines()
    map.drawparallels(np.arange(-90., 90., 10.), linewidth=2, labels=[1,0,0,0])
    map.drawmeridians(np.arange(-180., 180., 10.), linewidth=2, labels=[0,0,0,1])
    h = map.contourf(lon,lat,magnitude.T,levels=np.linspace(0,3,100))
    q = map.quiver(lon1,lat1,zonal_current.T,meridional_current.T,width=0.001, color='white', scale=150)    
    cbar = map.colorbar(h)
    cbar.set_label("Magnitude of current value in m/sec")
    plt.title("Currents (zonal and meridional) in Indian Ocean on {}".format(date.strip("\"")))

    return q, h


fig = plt.figure(figsize=(16,8))

with writer.saving(fig, "writer_test.mp4", dpi=250):
    for f in meridional_current_data_files:
        f = f[23:]
        update(f)
        writer.grab_frame()
