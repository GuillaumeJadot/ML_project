import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
    data_df = data_df.drop(['Sensor ID', 'Index'], axis=1)
    return data_df


def plot_data(data_df):
    # Create a subplot with 3 rows and 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(9, 9))

    # Scatter plot of Temperature data
    data_df.plot(kind='scatter', x='M.Time[d]', y='Temperature', alpha=0.8, c='R [m]', cmap='coolwarm', ax=axs[0, 0])

    data_df_clean = data_df[(data_df['Temperature'] < 1000)].copy()
    data_df_clean.plot(kind='scatter', x='M.Time[d]', y='Temperature', alpha=0.8, c='R [m]', cmap='coolwarm', ax=axs[0, 1])

    # Distribution of R [m]
    axs[0, 2].hist(data_df['R [m]'], bins=50, color='green')
    axs[0, 2].set_title('Distribution of R [m] [Train data]')

    # Filter rows where Temperature is greater than 120
    data_df = data_df[data_df['Temperature'] < 120]

    # Temperature vs radius
    radius = data_df['R [m]']
    temperature = data_df['Temperature']

    # Create the scatter plot
    axs[1, 0].scatter(radius, temperature, alpha=0.8, color='green', s=10)
    axs[1, 0].set_xlabel('Radius (m)')
    axs[1, 0].set_ylabel('Temperature')
    axs[1, 0].set_title('Temperature vs. Radius')

    # Relationship between material and temperature
    sns.barplot(data=data_df, x='Material', y='Temperature', ax=axs[1, 1])
    axs[1, 1].set_xlabel('Material')
    axs[1, 1].set_ylabel('Temperature')
    axs[1, 1].set_title('Relationship between Material and Temperature')

    # Count the number of points for each material
    material_counts = data_df['Material'].value_counts()

    # Plot the count of points for each material
    sns.barplot(x=material_counts.index, y=material_counts.values, ax=axs[1, 2])
    axs[1, 2].set_xlabel('Material')
    axs[1, 2].set_ylabel('Number of Points')
    axs[1, 2].set_title('Number of Points for Each Material')

    # To prevent overlapping of plots
    plt.tight_layout()

    # Display all the plots
    plt.show()


if __name__ == '__main__':
    data_df = load_and_process_data()
    plot_data(data_df)