# Import data
import time
import numpy as np

file_path = "data/datathon2_data/OneDrive_1_12-09-2020/PotentialTemperature_3D/001_29_Dec_2003.txt"

idx = 0

BAD_FLAG = '-1.E+34'

date = ""
LAT = set()
LON = set()
DEP = set()
SALT = []

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
                LAT.add(lat)
                LON.add(lon)
                DEP.add(dep)
                SALT.append(salt)


        else:
            break


        idx += 1

MAX_SALT = np.nanmax(SALT)
MIN_SALT = np.nanmin(SALT)

LAT = list(LAT)
LAT.sort()
LON = list(LON)
LON.sort()
DEP = list(DEP)
DEP.sort()

MIN_LAT, MAX_LAT = min(LAT),max(LAT)
MIN_LON, MAX_LON = min(LON),max(LON)

r,c = len(LON),len(LAT)

def getSaltForDepth(depth):
    salt = []
    for x in LON:
        arr = []
        for y in LAT:
            arr.append(OCEAN[(y,x,depth)])
        salt.append(np.array(arr))
    return np.array(salt)

# Define frames
import plotly.graph_objects as go

fig = go.Figure(frames=[go.Frame(data=go.Surface(
    z=dep * np.ones((r, c)),
    surfacecolor=getSaltForDepth(dep).T,
    cmin=MIN_SALT, cmax=MAX_SALT,
    colorbar_title="Potential Temperature (degree Celcius)",
    colorscale=[[0, 'white'],
                [0.01, 'white'],
                [0.01, 'red'],
                [1, 'yellow']],
                # [1, 'green']]
    ),
    name=str(dep)
    ) for dep in DEP])

# Add data to be displayed before animation starts
fig.add_trace(go.Surface(
    z=5.0 * np.ones((r, c)),
    surfacecolor=getSaltForDepth(5.0).T,
    cmin=MIN_SALT, cmax=MAX_SALT,
    colorbar_title="Potential Temperature (degree Celcius)",
    colorscale=[[0, 'white'],
                [0.01, 'red'],
                [1, 'yellow']],
                # [1, 'green']]
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
         title='Indian Ocean Potential Temperature with variation in depth in meters (z-direction) on 29 December 2003',
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