import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Indian Ocean - SSHA', artist='Swasti',
                comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)

# List of files to be visualised
ssha_data_files = glob.glob("data/ssha/*.txt")
ssha_data_files.sort()

BAD_FLAG = '-1.E+34'

def update(ssha_file, first_frame):
    idx = 0

    # Data structure to store the value of SSHA at location (LON,LAT)
    OCEAN = dict()
    date = ""

    with open(ssha_file,'r') as f:
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
    lon, lat = np.meshgrid(LON, LAT)
    h = plt.contourf(SSHA.T,levels = 100)
    if first_frame:
        plt.colorbar()
    plt.title("Indian Ocean SSHA on {}".format(date.strip("\"")))

    return h


fig = plt.figure(figsize=(16,8))

with writer.saving(fig, "writer_test.mp4", dpi=100):
    FIRST_FRAME = True
    for f in ssha_data_files:
        update(f, FIRST_FRAME)
        writer.grab_frame()
        FIRST_FRAME = False
