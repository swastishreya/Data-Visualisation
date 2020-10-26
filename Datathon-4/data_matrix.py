import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn

from traveling_sales_person import TravelingSalesPerson
from optimal_leaf_ordering import OptimalLeafOrdering

"""Data file paths"""
deaths = "../Datathon-3/time_series_covid_19_deaths.csv"
recovered = "../Datathon-3/time_series_covid_19_recovered.csv"
confirmed = "../Datathon-3/time_series_covid_19_confirmed.csv"

def read_data(filename):
    df_in = pd.read_csv(filename)
    dict_date = {date:'sum' for date in list(df_in.columns[4:])}
    df_in = df_in.groupby(['Country/Region']).agg(dict_date).reset_index()
    assert df_in.isnull().values.any() == False, "Dataframe has null values"
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
    corr_matrix = corr_matrix.fillna(0)

    # Change the range of similarity values
    corr_matrix = 1 + corr_matrix

    return corr_matrix

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
    list_country1, list_country2, list_d = [None] * n_lines, [None] * n_lines, [None] * n_lines

    line_index = 0
    for i in range(0, df_in.shape[0] - 1):
        for j in range(i + 1, df_in.shape[0]):
            index_i, index_j = df_in.index[i], df_in.index[j]
            list_country1[line_index] = df_in.at[index_i, 'Country/Region']
            list_country2[line_index] = df_in.at[index_j, 'Country/Region']
            diff_time = df_in.at[index_i, 'avg_time'] - df_in.at[index_j,'avg_time']
            list_d[line_index] = abs(diff_time)
            line_index += 1

    print(list_d, len(list_d))
    countries = list(set(list_country1+list_country2))
    countries = {countries[i]:i for i in range(len(countries))}
    # print(countries)

    matrix = np.full((len(countries), len(countries)),100)
    for i in range(len(countries)):
        matrix[i,i] = 0

    idx = 0
    for i,j in zip(list_country1,list_country2):
        if np.isnan(list_d[idx]):
            list_d[idx] = 100
        matrix[countries[i],countries[j]] = list_d[idx]
        matrix[countries[j],countries[i]] = list_d[idx]
        idx += 1

    return matrix


"""Average time matrix"""
df_deaths = read_data(deaths)
# deaths_matrix = get_average_time(df_deaths)

"""Correlation matrix"""
invert_df = get_transpose(df_deaths)
corr_edges = get_correlation_edges(invert_df)

"""Choose which matrix to visualize"""
data = corr_edges
# data = deaths_matrix

tsp = TravelingSalesPerson(data, data_type='data')
# olo = OptimalLeafOrdering(pd.DataFrame(data), data_type='data', metric='euclidean', method='complete')
seaborn.heatmap(data, cmap = cm.Blues,xticklabels=True, yticklabels=True)
plt.figure()

# Visualize the output data
Y = tsp.get_ordered_data()
# Y = olo.get_ordered_data()
seaborn.heatmap(Y, cmap = cm.Blues,xticklabels=True, yticklabels=True)
plt.show()

