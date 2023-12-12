import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_and_process_data():
    # Load data
    temp_df = pd.read_csv("TrainingData/Training-data-temperature.csv")
    humidity_df = pd.read_csv("TrainingData/Training-data-humidity.csv")
    pressure_df = pd.read_csv("TrainingData/Training-data-pressure.csv")
    coords_df = pd.read_csv("TrainingData/Coordinates-Training.csv")
    # Melt the data frames to long format
    temp_df = temp_df.melt(id_vars=['M.Time[d]'], var_name='Sensor ID', value_name='Temperature')
    humidity_df = humidity_df.melt(id_vars=['M.Time[d]'], var_name='Sensor ID', value_name='Humidity')
    pressure_df = pressure_df.melt(id_vars=['M.Time[d]'], var_name='Sensor ID', value_name='Pressure')
    # Merge the data frames
    data_df = temp_df.merge(humidity_df, on=['M.Time[d]', 'Sensor ID'])
    data_df = data_df.merge(pressure_df, on=['M.Time[d]', 'Sensor ID'])
    data_df = data_df.merge(coords_df, on=['Sensor ID'])
    # Drop unnecessary columns
    data_df = data_df.drop(['Sensor ID', 'Index', 'Material'], axis=1)
    return data_df


def filter_and_standardize_data(data_df):
    # Filter rows
    data_df = data_df[data_df['Temperature'] < 120]
    # Standardizing
    features = [col for col in data_df.columns if col != 'Temperature']
    scaler = StandardScaler()
    #data_df[features] = scaler.fit_transform(data_df[features])
    data_df.loc[:, features] = scaler.fit_transform(data_df[features])

    return data_df, scaler

def impute_data(data_df):
    # Apply the imputer to the DataFrame
    imputer = KNNImputer(n_neighbors=5)
    data_df_imputed = imputer.fit_transform(data_df)

    # Convert the result back to a DataFrame
    data_df = pd.DataFrame(data_df_imputed, columns=data_df.columns)
    
    return data_df

def run_preprocessing():
    data_df = load_and_process_data()
    data_df, scaler = filter_and_standardize_data(data_df)
    data_df = impute_data(data_df)
    
    return data_df, scaler

