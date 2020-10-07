import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.basemap import Basemap

deaths = 'time_series_covid_19_deaths.csv'
recovered = 'time_series_covid_19_recovered.csv'

def read_data(filename):
    df_in = pd.read_csv(filename)
    dict_date = {date:'sum' for date in list(df_in.columns[4:54])}
    df_in = df_in.groupby(['Country/Region']).agg(dict_date).reset_index()
    print(df_in.columns)
    return df_in

def get_lat_lon(filename):
    df_in = pd.read_csv(filename)
    df_lat_lon = df_in.iloc[:,:4]
    # dict_lat_lon = {val:'mean' for val in list(df_lat_lon.columns[2:])}
    # df_lat_lon = df_lat_lon.groupby(['Country/Region']).agg(dict_lat_lon).reset_index()
    df_lat_lon = df_lat_lon.groupby(['Country/Region']).nth(-1).reset_index()
    return df_lat_lon

def get_transpose(df_in):
    invert_columns = df_in['Country/Region'].unique()
    invert_index = df_in.columns[1:]

    invert_df = pd.DataFrame(index=invert_index,columns=invert_columns)
    invert_df = invert_df.fillna(0)
    for i in tqdm(df_in['Country/Region']):
        for j in invert_index:
            invert_df.at[j,i] += list(df_in[df_in['Country/Region'] == i][j])[0]
            
    return invert_df

def get_correlation_edges(invert_df):
    corr_matrix = invert_df.corr()
    countries = corr_matrix.index.values
    corr_matrix = np.asmatrix(corr_matrix)

    edges = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix[i])):
            if pd.isnull(corr_matrix[i][j]) == False:
                edges.append((countries[i],countries[j],{'Weight':corr_matrix[i][j]}))
            
    return edges

def get_graph_from_edges(edges):
    G = nx.Graph(edges)
    return G

def get_node_weight(df_in):
    num_cases = {}
    for index in df_in.index:
        num_cases[df_in['Country/Region'][index]] = float(df_in['3/11/20'][index])/50
    
    return num_cases

def get_average_time(df_in):
    dates_vec = list(df_in.columns)[1:]
    average_time_array = [None] * df_in.shape[0]

    for i, row in enumerate(df_in.index):

        weighted_sum, total_cases = 0, 0
        
        for j, date in enumerate(dates_vec):
            current = df_in.at[row, date]
            weighted_sum += j * current
            total_cases += current
        
        average_time_array[i] = weighted_sum / total_cases
        
    df_in['avg_time'] = average_time_array

    n_lines = int((df_in.shape[0] * (df_in.shape[0] - 1)) / 2)
    list_country1, list_country2, list_w, list_d = [None] * n_lines, [None] * n_lines, [None] * n_lines, [None] * n_lines

    line_index = 0
    epsilon = 0.001
    for i in range(0, df_in.shape[0] - 1):
        for j in range(i + 1, df_in.shape[0]):
            index_i, index_j = df_in.index[i], df_in.index[j]
            list_country1[line_index] = df_in.at[index_i, 'Country/Region']
            list_country2[line_index] = df_in.at[index_j, 'Country/Region']
            diff_time = df_in.at[index_i, 'avg_time'] - df_in.at[index_j,'avg_time']
            list_w[line_index] = (1 / (abs(diff_time) + epsilon))
            list_d[line_index] = abs(diff_time)
            line_index += 1
                    
    df_graph = pd.DataFrame(dict(
        Country1 = list_country1,
        Country2 = list_country2,
        Weight = list_w
    ))

    df_graph = df_graph.dropna(axis=0)

    return df_graph

def get_graph_from_df(df_graph):
    G = nx.from_pandas_edgelist(df_graph, 'Country1', 'Country2', 'Weight')
    return G

def get_sparse_graph(graph, num_cases, min_threshold, max_threshold):
    sparse_graph = nx.Graph(((u, v, e) for u,v,e in graph.edges(data=True) if e['Weight'] > min_threshold and e['Weight'] < max_threshold))
    sparse_vertex = set()
    for (u,v) in  sparse_graph.edges():
        sparse_vertex.add(u)
        sparse_vertex.add(v)
    sparse_vertex = list(sparse_vertex)
    sparse_vertex.sort()
    vertex_attributes = {u:num_cases[u] for u in sparse_vertex}

    return sparse_graph, vertex_attributes

def get_graph_stats(G, cases_dict):
    graph_dict = {(u, v): e for (u,v,e) in G.edges(data=True)}
    weights = [e['Weight'] for (u,v,e) in G.edges(data=True)]
    print("Maximum weight: ", max(weights))
    print("Minimum weight: ", min(weights))
    print("Mean weight: ", np.mean(weights))

    top_10_cases = list(zip(cases_dict.values(),cases_dict.keys()))
    top_10_cases.sort(reverse=True)
    top_10_cases = top_10_cases[:10]
    return graph_dict, top_10_cases

df_deaths = read_data(deaths)
df_country = get_lat_lon(deaths)
num_deaths = get_node_weight(df_deaths)
df_death_graph = get_average_time(df_deaths)
deaths_graph = get_graph_from_df(df_death_graph)

"""Sparse death graph"""

# sparse_deaths_graph, death_attributes = get_sparse_graph(deaths_graph, num_deaths, 10, 100)
# e_color = sorted([e['Weight'] for u,v,e in sparse_deaths_graph.edges(data=True)])
# nx.draw(sparse_deaths_graph, with_labels=True,pos=positions,edge_color=e_color, width=3, edge_cmap=plt.cm.hot, nodelist=death_attributes.keys(), node_size=[v*10 for v in death_attributes.values()])

"""Top 10 graph"""
graph_d, top_10 = get_graph_stats(deaths_graph, num_deaths)
edges = []
death_attributes = {v:w for w,v in top_10}
for i in range(10):
    for j in range(i+1, 10):
        if (top_10[i][1],top_10[j][1]) in graph_d:
            edges.append((top_10[i][1],top_10[j][1],graph_d[top_10[i][1],top_10[j][1]]))
        elif (top_10[j][1],top_10[i][1]) in graph_d:
            edges.append((top_10[j][1],top_10[i][1],graph_d[top_10[j][1],top_10[i][1]]))

sparse_deaths_graph = nx.Graph(edges)
sparse_deaths_graph, death_attributes = get_sparse_graph(sparse_deaths_graph, num_deaths, 0, 100)
e_color = sorted([e['Weight'] for u,v,e in sparse_deaths_graph.edges(data=True)])
# nx.draw(sparse_deaths_graph, with_labels=True,nodelist=death_attributes.keys(), node_size=[v*10 for v in death_attributes.values()])


# Set up base map
plt.figure(figsize=(15,20))
m = Basemap(
        projection='merc',
        llcrnrlon=-180,
        llcrnrlat=-60,
        urcrnrlon=180,
        urcrnrlat=70,
        lat_ts=0,
        resolution='l',
        suppress_ticks=True)

# import long lat as m attribute
mx, my = m(df_country['Long'].values, df_country['Lat'].values)
pos = {}
for idx, elem in enumerate (df_country['Country/Region']):
    pos[elem] = (mx[idx], my[idx])

# draw nodes and edges and over aly on basemap
nx.draw_networkx_nodes(G = sparse_deaths_graph, pos = pos, node_list = sparse_deaths_graph.nodes(), node_color = 'r', alpha = 0.6,
                        node_size = [num_deaths[s]*5 for s in sparse_deaths_graph.nodes()], with_labels=True)
nx.draw_networkx_edges(G = sparse_deaths_graph, pos = pos, edge_color=e_color, width=2, edge_cmap=plt.cm.hot,
                        alpha=0.6, arrows = False)
nx.draw_networkx_labels(G = sparse_deaths_graph, pos = pos, font_size=15, font_color='y',
		                labels = {x:x for x in sparse_deaths_graph.nodes() if num_deaths[x] >= 0})

m.drawcountries(linewidth = 1)
m.drawstates(linewidth = 0.2)
m.drawcoastlines(linewidth=1)
plt.tight_layout()
sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(vmin = min(e_color), vmax=max(e_color)))
cbar = plt.colorbar(sm)
cbar.set_label("Edge weights")
plt.title("Top 10 countries with maximum deaths due to COVID-19 by the end of 11/3/20")
plt.show()