import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from Data_preprocessing import run_preprocessing


class ThreeLayerNet(nn.Module):
    """
    Neural network model class. 
    """
    def __init__(self, input_size, hidden_size, middle_size, output_size):
        """
        Initialize the model by setting up the layers.
        """
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, middle_size)
        self.fc3 = nn.Linear(middle_size, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(middle_size)
        self.bn3 = nn.BatchNorm1d(32)

        # Apply Kaiming (He) initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')

    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        out = self.fc4(x)
        return out



def split_and_prepare_data(data_df):
    """
    Splits the data into train and test sets and prepares them for use in the model.
    """
    X_scaled = data_df.drop('Temperature', axis=1)
    y = data_df['Temperature']
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_nn_model(X_train_scaled, y_train, input_size, hidden_size, middle_size, output_size, epochs):
    """
    Trains the model and returns the trained model along with the training losses.
    """
    X_train_scaled = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    model = ThreeLayerNet(input_size, hidden_size, middle_size, output_size)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.88)
    train_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_scaled)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        scheduler.step()
    return model, train_losses


def evaluate_nn_model(model, X_test_scaled, y_test):
    """
    Evaluates the model on the test set.
    """
    model.eval()
    X_test_scaled = torch.tensor(X_test_scaled.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)
    predictions = model(X_test_scaled).detach().numpy()
    mse_neural = mean_squared_error(y_test, predictions)
    mae_neural = mean_absolute_error(y_test, predictions)
    return predictions, mse_neural, mae_neural


def import_and_prepare_new_data():
    """
    Imports new data and prepares it for prediction.
    """
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



def predict_on_new_data(model, data_new):
    """
    Makes predictions on new data.
    """
    model.eval()  # put the model in evaluation mode
    # Prepare the input data
    X_new_scaled = scaler.transform(data_new)
    X_new_scaled = torch.tensor(X_new_scaled, dtype=torch.float32)

    # Make predictions using the new data
    y_pred_new = model(X_new_scaled)

    # Add the predictions to the new data DataFrame
    data_new['Predicted Temperature'] = y_pred_new.detach().numpy()
    return data_new, y_pred_new


def create_and_save_csv(data_new, humidity_new):
    """
    Creates and saves the predicted temperatures as a CSV file.
    """
    unique_time_steps = humidity_new["M.Time[d]"][0:32]
    column_names = ['M.Time[d]'] + [f'N_{i}' for i in range(901, 1047)]
    predicted_temp_new_df = pd.DataFrame(np.nan, index=np.arange(len(unique_time_steps)), columns=column_names)
    for index, row in data_new.iterrows():
        time_indices = np.where(unique_time_steps == row['M.Time[d]'])[0]
        if time_indices.size > 0:
            time_index = time_indices[0]
            sensor_index = int(row['Unnamed: 0']) + 901
            if 901 <= sensor_index <= 1046:
                predicted_temp_new_df.loc[time_index, f'N_{sensor_index}'] = row['Predicted Temperature']
    predicted_temp_new_df['M.Time[d]'] = unique_time_steps   
    predicted_temp_new_df = predicted_temp_new_df.drop('N_909', axis=1)
    predicted_temp_new_df = predicted_temp_new_df.rename(columns={'M.Time[d]': ''})
    predicted_temp_new_df.to_csv('NN_model_results.csv', index=False)




def run_nn_model(data_df, input_size, hidden_size, middle_size, output_size, epochs):
    """
    Main function to run the entire model.
    """
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_prepare_data(data_df)
    model, train_losses = train_nn_model(X_train_scaled, y_train, input_size, hidden_size, middle_size, output_size, epochs)
    predictions, mse_neural, mae_neural = evaluate_nn_model(model, X_test_scaled, y_test)
    data_new, humidity_new = import_and_prepare_new_data()
    data_new, y_pred_new = predict_on_new_data(model, data_new)
    create_and_save_csv(data_new, humidity_new)
    return predictions, mse_neural, mae_neural, data_new, y_pred_new, train_losses


if __name__ == '__main__':
    # Load and preprocess the data
    data_df, scaler = run_preprocessing()   
    # Run the RF model with the preprocessed data
    run_nn_model(data_df, 8, 128, 64, 1, 2000)
