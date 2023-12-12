# import necessary libraries
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


from Data_preprocessing import run_preprocessing


def split_data(data_df):
    # Features (Time, Humidity, Pressure, Radius)
    X_scaled = data_df.drop('Temperature', axis=1)
    # Target (Temperature)
    y = data_df['Temperature']
    # Split the dataset into training and testing sets
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_random_forest(X_train_scaled, y_train):
    # Initialize the model
    rf = RandomForestRegressor(n_estimators=15, random_state=42)
    # Train the model
    rf.fit(X_train_scaled, y_train)
    return rf

def predict_and_evaluate(rf, X_test_scaled, y_test):
    # Make predictions on the test set
    y_pred = rf.predict(X_test_scaled)
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return y_pred, mae, rmse, r2

def import_and_prepare_new_data():
    # Read the new data
    humidity_new = pd.read_csv("TestDataStudent/Test-Time-humidity.csv")
    pressure_new = pd.read_csv("TestDataStudent/Test-Time-pressure.csv")
    coords_new = pd.read_csv("TestDataStudent/Coordinates-Test.csv")
    # Melt the data frames to long format
    humidity_new = humidity_new.melt(id_vars=['M.Time[d]'], var_name='Sensor ID', value_name='Humidity')
    pressure_new = pressure_new.melt(id_vars=['M.Time[d]'], var_name='Sensor ID', value_name='Pressure')
    # Merge the data frames
    data_new = humidity_new.merge(pressure_new, on=['M.Time[d]', 'Sensor ID'])
    data_new = data_new.merge(coords_new, on=['Sensor ID'])
    # Drop unnecessary columns
    data_new = data_new.drop(['Sensor ID', 'Index', 'Material'], axis=1)
    return data_new, humidity_new

def predict_on_new_data(rf, scaler, data_new):
    # Prepare the input data
    X_new_scaled = scaler.transform(data_new)
    # Make predictions using the new data
    y_pred_new = rf.predict(X_new_scaled)
    # Add the predictions to the new data DataFrame
    data_new['Predicted Temperature'] = y_pred_new
    return data_new, y_pred_new

def create_and_save_csv(data_new, humidity_new):
    # Calculate the unique time steps, ensuring all 32 time steps are included
    unique_time_steps = humidity_new["M.Time[d]"][0:32]
    # Create an empty DataFrame with the desired shape (num_time_steps, 146)
    column_names = ['M.Time[d]'] + [f'N_{i}' for i in range(901, 1047)]
    predicted_temp_new_df = pd.DataFrame(np.nan, index=np.arange(len(unique_time_steps)), columns=column_names)
    # Fill the predicted_temp_new_df DataFrame with the predicted temperature values
    for index, row in data_new.iterrows():
        time_indices = np.where(unique_time_steps == row['M.Time[d]'])[0]
        if time_indices.size > 0:
            time_index = time_indices[0]
            sensor_index = int(row['Unnamed: 0']) + 901
            if 901 <= sensor_index <= 1046:  # Check if the sensor index is within the valid range
                predicted_temp_new_df.loc[time_index, f'N_{sensor_index}'] = row['Predicted Temperature']

    # Fill the 'M.Time[d]' column with the unique time values
    predicted_temp_new_df['M.Time[d]'] = unique_time_steps   
    #We drop the column of sensors 909 because these data are missing, even in the data
    predicted_temp_new_df = predicted_temp_new_df.drop('N_909', axis=1)
    # replace the name of the M.Time[d]
    predicted_temp_new_df = predicted_temp_new_df.rename(columns={'M.Time[d]': ''})
    # Save the predictions to a .csv file
    predicted_temp_new_df.to_csv('RF_model_results.csv', index=False)

def run_rf_model(data_df):
    data_df, scaler = run_preprocessing()
    X_train_scaled, X_test_scaled, y_train, y_test = split_data(data_df)
    rf = train_random_forest(X_train_scaled, y_train)
    y_pred, mae, rmse, r2 = predict_and_evaluate(rf, X_test_scaled, y_test)
    data_new, humidity_new = import_and_prepare_new_data()
    data_new, y_pred_new = predict_on_new_data(rf, scaler, data_new)
    create_and_save_csv(data_new, humidity_new)
    return y_pred, mae, rmse, r2, data_new, y_pred_new



if __name__ == '__main__':
    # Load and preprocess the data
    data_df = run_preprocessing()
    # Run the RF model with the preprocessed data
    run_rf_model(data_df)

