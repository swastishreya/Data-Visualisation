import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

df_in = pd.read_csv('time_series_covid_19_deaths.csv')
dict_date = {date:'sum' for date in list(df_in.columns[4:])}
df_in = df_in.groupby(['Country/Region']).agg(dict_date).reset_index()
df_in.iloc[:,:].head()

num_deaths = {}
for index in df_in.index:
    num_deaths[df_in['Country/Region'][index]] = float(df_in['9/23/20'][index])/246


dates_vec = list(df_in.columns)[1:]
average_time_vec = [None] * df_in.shape[0]

for i, row_index in enumerate(df_in.index):

    weighted_sum, total_deaths = 0, 0
    
    for j, date in enumerate(dates_vec):
        current_term = df_in.at[row_index, date]
        weighted_sum += j * current_term
        total_deaths += current_term
    
    average_time_vec[i] = weighted_sum / total_deaths
    
df_in['avg_time'] = average_time_vec

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
    Weight = list_w,
    Distance = list_d
))


df_graph = df_graph.dropna(axis=0)
df_graph.to_csv('df_graph.csv', index=False)


covid_graph = nx.from_pandas_edgelist(df_graph, 'Country1', 'Country2', 'Weight')
sparse_covid_graph = nx.Graph(((u, v, e) for u,v,e in covid_graph.edges(data=True) if e['Weight'] > 0.0 and e['Weight'] < 0.0150))
sparse_vertex = set()
for (u,v) in  sparse_covid_graph.edges():
    sparse_vertex.add(u)
    sparse_vertex.add(v)
sparse_vertex = list(sparse_vertex)
sparse_vertex.sort()
d = [e['Weight'] for u,v,e in covid_graph.edges(data=True)]
print(np.mean(np.array(d)))
# nx.draw(sparse_covid_graph,with_labels=True)

vertex_attributes = {u:num_deaths[u] for u in sparse_vertex}
nx.draw(sparse_covid_graph, with_labels=True,nodelist=vertex_attributes.keys(), node_size=[v*100 for v in vertex_attributes.values()])
plt.show()
