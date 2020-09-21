# Import data
import time
import numpy as np

meridional_curr_file_path = "data/datathon2_data/OneDrive_1_12-09-2020/meridional-current_3D/063_04_Nov_2004.txt"
zonal_curr_file_path = "data/datathon2_data/OneDrive_1_12-09-2020/zonal-current_3D/063_04_Nov_2004.txt"

idx = 0

BAD_FLAG = '-1.E+34'

date = ""
LAT = set()
LON = set()
DEP = set()
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
                LAT.add(lat)
                LON.add(lon)
                DEP.add(dep)


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
                LAT.add(lat)
                LON.add(lon)
                DEP.add(dep)
                if OCEAN[(lat,lon,dep)][0] == np.nan or OCEAN[(lat,lon,dep)][1] == np.nan:
                    mag = np.nan
                else:
                    mag = np.sqrt(OCEAN[(lat,lon,dep)][0]**2 + OCEAN[(lat,lon,dep)][1]**2)
                OCEAN[(lat,lon,dep)] = mag
                MAGNITUDE.append(mag)


        else:
            break


        idx += 1

MAX_MAG = np.nanmax(MAGNITUDE)
MIN_MAG = np.nanmin(MAGNITUDE)

LAT = list(LAT)
LAT.sort()
LON = list(LON)
LON.sort()
DEP = list(DEP)
DEP.sort()

MIN_LAT, MAX_LAT = min(LAT),max(LAT)
MIN_LON, MAX_LON = min(LON),max(LON)

r,c = len(LON),len(LAT)

def getMagnitudeForDepth(depth):
    mag = []
    for x in LON:
        arr = []
        for y in LAT:
            if (y,x,depth) not in OCEAN or type(OCEAN[(y,x,depth)]) == list:
                arr.append(np.nan)
            else:
                arr.append(OCEAN[(y,x,depth)])
        mag.append(np.array(arr))
    mag = np.array(mag)
    mag[np.isnan(mag)] = -100
    return mag

# Define frames
import plotly.graph_objects as go

fig = go.Figure(frames=[go.Frame(data=go.Surface(
    z=dep * np.ones((r, c)),
    surfacecolor=getMagnitudeForDepth(dep).T,
    cmin=MAX_MAG - MAX_MAG/0.99, cmax=MAX_MAG,
    colorbar_title="Magnitude of currents (m/sec)",
    colorscale=[[0, 'white'],
                [0.01, 'white'],
                [0.01, 'blue'],
                [1, 'red']]
    ),
    name=str(dep)
    ) for dep in DEP])

# Add data to be displayed before animation starts
fig.add_trace(go.Surface(
    z=225.0 * np.ones((r, c)),
    surfacecolor=getMagnitudeForDepth(5.0).T,
    cmin=MAX_MAG - MAX_MAG/0.99, cmax=MAX_MAG,
    colorbar_title="Magnitude of currents (m/sec)",
    colorscale=[[0, 'white'],
                [0.01, 'white'],
                [0.01, 'blue'],
                [1, 'red']]
    ))

def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

# Layout
fig.update_layout(
         title='Indian Ocean Magnitude of zonal and meridional currents with variation in depth in meters (z-direction) on 4 November 2004',
         width=1200,
         height=800,
         scene=dict(
                    zaxis=dict(range=[5.0, 225.0],autorange=False,title='Depth in meters'),
                    xaxis = dict(title='Longitude'),
                    yaxis = dict(title='Latitude'),
                    aspectratio=dict(x=1.5, y=1, z=1),
                    ),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
         ],
         sliders=sliders
)

fig.show()