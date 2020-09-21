import numpy as np
import plotly.graph_objects as go

file_path = "data/datathon2_data/OneDrive_1_12-09-2020/Salinity_3D/001_29_Dec_2003.txt"

idx = 0

BAD_FLAG = '-1.E+34'

date = ""
LAT = []
LON = []
DEP = []
SALT = [] # Note that this can be changed to any variable 

OCEAN = dict()

with open(file_path,'r') as f:
    while(f):
        r = f.readline()
        if r != '':
            if idx >= 11:
                data = r.strip().split(',')
                date = data[0]
                lon = float(data[2])
                lat = float(data[3])
                dep = float(data[4])
                salt = data[5]
                if salt == BAD_FLAG:
                    salt = np.nan
                else:
                    salt = float(salt)
                OCEAN[(lat,lon,dep)] = salt
                if dep > 50:
                    continue
                LAT.append(lat)
                LON.append(lon)
                DEP.append(dep)
                SALT.append(salt)


        else:
            break


        idx += 1

MAX_SALT = np.nanmax(SALT)
MIN_SALT = np.nanmin(SALT)

fig= go.Figure(
    data=go.Isosurface(
        x=LON,
        y=LAT,
        z=DEP,
        value=SALT,
        isomin=MIN_SALT,
        isomax=MAX_SALT,
        surface_count=50,
        colorbar_nticks=5,
        colorbar_title="Salinity (psu)",
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
            text='Indian Ocean Salinity with variation in depth in meters (z-direction) on 29 December 2003'
        )
    )
)

fig.show()
