import numpy as np
import plotly.graph_objects as go

meridional_curr_file_path = "data/datathon2_data/OneDrive_1_12-09-2020/meridional-current_3D/063_04_Nov_2004.txt"
zonal_curr_file_path = "data/datathon2_data/OneDrive_1_12-09-2020/zonal-current_3D/063_04_Nov_2004.txt"

idx = 0

BAD_FLAG = '-1.E+34'

date = ""
LAT = []
LON = []
DEP = []
MAGNITUDE = []

OCEAN = dict()

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
                meridional_current = data[5]
                if meridional_current == BAD_FLAG:
                    meridional_current = np.nan
                else:
                    meridional_current = float(data[5])*-1
                OCEAN[(lat,lon,dep)] = [0,meridional_current]

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
                date = data[0]
                lon = float(data[2])
                lat = float(data[3])
                dep = float(data[4])
                zonal_current = data[5]
                if zonal_current == BAD_FLAG:
                    zonal_current = np.nan
                else:
                    zonal_current = float(data[5])*-1
                OCEAN[(lat,lon,dep)][0] = zonal_current
                mag = np.nan
                if OCEAN[(lat,lon,dep)][0] == np.nan or OCEAN[(lat,lon,dep)][1] == np.nan:
                    mag = np.nan
                else:
                    mag = np.sqrt(OCEAN[(lat,lon,dep)][0]**2 + OCEAN[(lat,lon,dep)][1]**2)
                OCEAN[(lat,lon,dep)] = mag
                if dep > 50:
                    continue
                LAT.append(lat)
                LON.append(lon)
                DEP.append(dep)
                MAGNITUDE.append(mag)


        else:
            break


        idx += 1

MAX_MAG = np.nanmax(MAGNITUDE)
MIN_MAG = np.nanmin(MAGNITUDE)

fig= go.Figure(
    data=go.Isosurface(
        x=LON,
        y=LAT,
        z=DEP,
        value=MAGNITUDE,
        isomin=MIN_MAG,
        isomax=MAX_MAG,
        surface_count=50,
        colorbar_nticks=10,
        colorscale="viridis",
        colorbar_title="Magnitude of currents (m/sec)",
        # opacity=0.7,
        caps=dict(x_show=False, y_show=False, z_show=False)
    ),
    layout=go.Layout(
        scene = dict(
                    xaxis = dict(title='Longitude'),
                    yaxis = dict(title='Latitude'),
                    zaxis = dict(title='Depth in meters'),
                ),
        title = go.layout.Title(
            text='Indian Ocean Magnitude of zonal and meridional currents with variation in depth in meters (z-direction) on 4 November 2004'
        )
    )
)

fig.show()
