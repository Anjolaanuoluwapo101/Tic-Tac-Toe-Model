import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
url = "datasets/tic-tac-toe-endgame.csv"  # Replace with the actual URL or file path
data = pd.read_csv(url)

#convert categorical data first to numeric format
def convert_categorical_to_numeric(data):
    # Initialize a LabelEncoder
    encoder = LabelEncoder()

    # Iterate through the first 9 columns
    for col in data.columns[:9]:
        data[col] = encoder.fit_transform(data[col])

    # Convert the 10th column
    data['V10'] = encoder.fit_transform(data['V10'])

    return data

data = convert_categorical_to_numeric(data)
data.to_csv('tictactoe_encoded2.csv', index=False)