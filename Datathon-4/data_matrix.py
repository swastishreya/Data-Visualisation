import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn

from traveling_sales_person import TravelingSalesPerson
from optimal_leaf_ordering import OptimalLeafOrdering

deaths = "../Datathon-3/time_series_covid_19_deaths.csv"
recovered = "../Datathon-3/time_series_covid_19_recovered.csv"

def read_data(filename):
    df_in = pd.read_csv(filename)
    dict_date = {date:'sum' for date in list(df_in.columns[4:])}
    df_in = df_in.groupby(['Country/Region']).agg(dict_date).reset_index()
    print(df_in.isnull().values.any())
    return df_in

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
    corr_matrix = corr_matrix.fillna(-1)
    countries = corr_matrix.index.values
    corr_matrix = corr_matrix.values
    corr_matrix = 1 - corr_matrix

    countries = list(zip(countries,range(len(countries))))
    print(countries)

    return corr_matrix

def get_node_weight(df_in):
    num_cases = {}
    for index in df_in.index:
        num_cases[df_in['Country/Region'][index]] = float(df_in['9/23/20'][index])/246
    
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

    countries = list(set(list_country1+list_country2))
    countries = {countries[i]:i for i in range(len(countries))}
    # print(countries)

    matrix = np.zeros((len(countries), len(countries)))
    idx = 0
    for i,j in zip(list_country1,list_country2):
        if np.isnan(list_d[idx]):
            print("kaka")
            list_d[idx] = 100
        matrix[countries[i],countries[j]] = list_d[idx]
        matrix[countries[j],countries[i]] = list_d[idx]
        idx += 1

    return matrix



df_deaths = read_data(deaths)
# num_deaths = get_node_weight(df_deaths)
deaths_matrix = get_average_time(df_deaths)

"""Sparse deaths correlation matrix"""
# invert_df = get_transpose(df_deaths)
# corr_edges = get_correlation_edges(invert_df)

# data = corr_edges
data = deaths_matrix

tsp = TravelingSalesPerson(data)
# olo = OptimalLeafOrdering(pd.DataFrame(data), metric='euclidean', method='complete')
seaborn.heatmap(data)
plt.figure()

# Visualize the output data
Y = tsp.get_ordered_data()
# Y = olo.get_ordered_data()
seaborn.heatmap(Y)
plt.show()

"""Top 10 graph"""
# graph_d, top_10 = get_graph_stats(deaths_graph, num_deaths)
# edges = []
# death_attributes = {v:w for w,v in top_10}
# for i in range(10):
#     for j in range(i+1, 10):
#         if (top_10[i][1],top_10[j][1]) in graph_d:
#             edges.append((top_10[i][1],top_10[j][1],graph_d[top_10[i][1],top_10[j][1]]))
#         elif (top_10[j][1],top_10[i][1]) in graph_d:
#             edges.append((top_10[j][1],top_10[i][1],graph_d[top_10[j][1],top_10[i][1]]))

