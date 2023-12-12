# Predicting Temperature Around Nuclear Waste Canisters

## Project Overview

This project aims to develop machine learning models capable of predicting the temperature at different sensor moments in a nuclear waste canister. Two models - a Random Forest model and a Neural Network model - have been trained on provided datasets to solve this regression problem.

### The repository contains four primary Python scripts:

1. `Data_preprocessing.py`: Prepares the data for training and testing the models.
2. `Data_preprocessing_plots.py`: Produces plots for data analysis.
3. `RF_model.py`: Trains and evaluates the Random Forest model.
4. `NN_model.py`: Trains and evaluates the Neural Network model.

## Getting Started

Follow the steps below to reproduce the results:

### Prerequisites

Ensure you have Python 3.8 or later installed. This project is implemented using Python and relies on several Python libraries.

### Procedure

In order to run the models and thus have the results of the predicted temperatures in a `.csv` file, please follow this procedure:

1. Download the codes and the data and make a folder with all of them.
2. Open your terminal.
3. Create a new conda environment:

> conda create --name groupeX_ML_results python=3.8
> conda activate groupeX_ML_results

4. Navigate to the project directory and install the required dependencies using pip:

> cd repository
> pip install -r requirements.txt


### Running the Scripts

5. Run the following command in the terminal to execute the main script:

> python name.py

Run the RF_model or the NN_model to have to result for each model. It will automatically run the data_prepocessing script.

After running the script, the predicted temperature results will be saved in a .csv file in your project directory.