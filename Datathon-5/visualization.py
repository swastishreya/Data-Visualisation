import pandas as pd
import numpy as np
import plotly.express as px

def viz_lifeExp_country(data, years=[i for i in range(2000,2015)]):
  data = data.loc[data['year'].isin(years)]
  data["life_expectancy_at_birth_all"] = (data["life_expectancy_at_birth_men"]*data["total_population_male"] + data["life_expectancy_at_birth_women"]*data["total_population_female"])/(data["total_population_male"] + data["total_population_female"])

  fig = px.sunburst(data, path=['country','year'], values='life_expectancy_at_birth_all',
                    color='life_expectancy_at_birth_all', hover_data=['life_expectancy_at_birth_all'],
                    color_continuous_scale='RdBu',)
                                        
  fig.show()

def viz_lifeExp_europe(data, years=[i for i in range(2000,2015)]):
  data = data.loc[data['year'].isin(years)]
  data = data.loc[data['country'].isin(['United Kingdom','Germany','France','Italy','Netherlands','Malta','Israel','Belgium','Russia'])]
  data['population_in_M_per_sq_km'] = data['total_population']/data['area_square_kilometres']

  fig = px.sunburst(data, path=['country','year'], values='population_in_M_per_sq_km',
                    color='population_in_M_per_sq_km', hover_data=['total_population'],)
                    # color_continuous_scale='RdBu',)
                                        
  fig.show()

def viz_lifeExp_female_fertility(data):
  # Female life expectancy
  fig = px.parallel_coordinates(data[["country_index","life_expectancy_at_birth_women","mean_age_of_women_at_birth_of_first_child","adolescent_fertility_rate","life_expectancy_at_age_65_women"]], color="life_expectancy_at_age_65_women", 
                              labels={"country_index":"country_index","adolescent_fertility_rate":"adolescent_fertility_rate","life_expectancy_at_birth_women": "life_expectancy_at_birth_women", "mean_age_of_women_at_birth_of_first_child": "mean_age_of_women_at_birth_of_first_child","life_expectancy_at_age_65_women":"life_expectancy_at_age_65_women",},
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
  fig.show()

def viz_computer_usage_employment(data, years=[i for i in range(2000,2015)]):
  data = data.loc[data['year'].isin(years)]

  data["life_expectancy_at_birth_all"] = (data["life_expectancy_at_birth_men"]*data["total_population_male"] + data["life_expectancy_at_birth_women"]*data["total_population_female"])/(data["total_population_male"] + data["total_population_female"])
  data["life_expectancy_at_age_65_all"] = (data["life_expectancy_at_age_65_men"]*data["total_population_male"] + data["life_expectancy_at_age_65_women"]*data["total_population_female"])/(data["total_population_male"] + data["total_population_female"])
  data["computer_use_16_24_all"] = data["computer_use_16_24_male"] + data["computer_use_16_24_female"]
  data["computer_use_25_54_all"] = data["computer_use_25_54_male"] + data["computer_use_25_54_female"]
  data["computer_use_55_74_all"] = data["computer_use_55_74_male"] + data["computer_use_55_74_female"]

  # Parallel coordinates plot
  columnsNew = ["life_expectancy_at_birth_all", "life_expectancy_at_age_65_all","computer_use_16_24_all", "computer_use_25_54_all", "computer_use_55_74_all", "youth_unemployment_rate", "unemployment_rate"]
  cols = {i:i for i in columnsNew}

  fig = px.parallel_coordinates(data[columnsNew], color="life_expectancy_at_age_65_all", 
                              labels=cols,
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)

  # # Scatter matrix plot
  # columnsNew = ["life_expectancy_at_birth_all", "computer_use_16_24_all","unemployment_rate"]
  # cols = {i:i for i in columnsNew}
  # fig = px.scatter_matrix(data[columnsNew],
  #   dimensions=columnsNew,
  #   color="unemployment_rate",
  #   title="Scatter matrix of UNECE employement data",
  #   labels=cols)
                          
  fig.show()

def viz_germany(data, years=[i for i in range(2000,2015)]):
  data = data.loc[data['year'].isin(years)]
  data = data.loc[data['country'] == 'Germany']
  columnsNew = ["year", "total_fertility_rate", "total_population"]
  cols = {i:i for i in columnsNew}

  fig = px.scatter_matrix(data[columnsNew],
    dimensions=columnsNew,
    color="total_population",
    title="Scatter matrix of UNECE Germany data",
    labels=cols)
                          
  fig.show()

def viz_employement(data):
  columnsNew = ["economic_activity_rate_men_15_64", "economic_acivity_rate_women_15_64", "unemployment_rate"]
  cols = {i:i for i in columnsNew}

  fig = px.scatter_matrix(data[columnsNew],
    dimensions=columnsNew,
    color="unemployment_rate",
    title="Scatter matrix of UNECE employement data",
    labels=cols)
                          
  fig.show()

def viz_gender_pay_gap(data, years=[i for i in range(2000,2015)]):
  data = data.loc[data['year'].isin(years)]
  data["world"] = "world"
  fig = px.treemap(data, path=['world','country','year'], values='gender_pay_gap_in_monthly_earnings',
                    color='gender_pay_gap_in_monthly_earnings', hover_data=['gender_pay_gap_in_monthly_earnings'],)
  fig.show()

if __name__ == '__main__':
  data = pd.read_csv("unece.csv",header=0)

  print(data.columns)

  country = list(set(data["country"]))
  country_index = {country[i]:i for i in range(len(country))}

  data["country_index"] = data.apply(lambda row: country_index[row.country], axis = 1) 
  data = data.fillna(data.mean())

  # viz_lifeExp_country(data)
  # viz_computer_usage_employment(data)
  # viz_employement(data)
  # viz_lifeExp_europe(data)
  # viz_gender_pay_gap(data)
  # viz_germany(data)
  # viz_lifeExp_female_fertility(data)


