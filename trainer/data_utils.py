from __future__ import print_function
import pandas as pd
import numpy as np
import re
from tensorflow.python.lib.io import file_io

# Cloud/local path to passenger train/predict files.
_PASSENGER_DATA_LOCAL = 'data/passengerTrain.csv'
_PASSENGER_PREDICT_DATA_LOCAL = 'data/passengerTest.csv'
_PASSENGER_DATA_CLOUD = 'gs://keras-titanic-models/data/passengerTrain.csv'
_PASSENGER_PREDICT_DATA_CLOUD = 'gs://keras-titanic-models/data/passengerTest.csv'


def get_dataset(cloud_train):
    print("\nImporting data... Please wait.")
    # Load correct cloud/local train/predict dataset path.
    if not cloud_train:
        data = _load_dataset("Training", _PASSENGER_DATA_LOCAL)
        predict_data = _load_dataset("Predict", _PASSENGER_PREDICT_DATA_LOCAL)
    else:
        gs_train = file_io.FileIO(_PASSENGER_DATA_CLOUD, mode='r')
        gs_predict = file_io.FileIO(_PASSENGER_PREDICT_DATA_CLOUD, mode='r')
        data = _load_dataset("Training", gs_train)
        predict_data = _load_dataset("Predict", gs_predict)
    return data, predict_data


# Loads our training and testing data.
def _load_dataset(name, filepath):
    # Specify CSV column dtypes. Reduces memory usage during import.
    dtypes = {'PassengerId': np.int32, 'Survived': np.int32, 'Pclass': np.int32, 'Name': str, 'Sex': str,
              'Age': np.float16, 'SibSp': np.int32, 'Parch': np.int32, 'Ticket': str,
              'Fare ': np.float16, 'Cabin': str, 'Embarked': str}

    # Read csv into pandas dataframe.
    df = pd.read_csv(filepath, dtype=dtypes, index_col=0)

    # Load engineered features into dataframe.
    df = _engineer_features(df)

    print("%s data imported successfully." % name)
    return df


def _engineer_features(df):
    # Create new feature FamilySize.
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Create new feature IsAlone from FamilySize.
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Fill nan Fare values.
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Extract name titles.
    def get_passenger_title(name):
        title_search = re.search(" ([A-Za-z]+)\.", name)
        # Extract and return Title if it exists.
        if title_search:
            return title_search.group(1)
        return ""

    # Create a new feature Title for passenger names.
    df['Title'] = df['Name'].apply(get_passenger_title)

    # Group uncommon titles.
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    # Replace title errors/uncommon titles.
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Map Title to buckets.
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    print(df)
    # Fill nan Age values. For men, each age is the average age for titles and classes.
    # For women, the ages are averaged by the miss/mrs titles and the average year married in 1912.
    df.loc[(df['Title'] == 2) & (df['Age'].isnull()), 'Age'] = 21
    df.loc[(df['Title'] == 3) & (df['Age'].isnull()), 'Age'] = 28
    df.loc[(df['Title'] == 4) & (df['Age'].isnull()), 'Age'] = 5
    df.loc[(df['Title'] == 1) & (df['Age'].isnull()) & (df['Pclass'] == 3) & (df['IsAlone'] == 1), 'Age'] = 29
    df.loc[(df['Title'] == 1) & (df['Age'].isnull()) & (df['Pclass'] == 3) & (df['IsAlone'] == 0), 'Age'] = 21
    df.loc[(df['Title'] == 1) & (df['Age'].isnull()) & (df['Pclass'] == 1) & (df['IsAlone'] == 1), 'Age'] = 45
    df.loc[(df['Title'] == 1) & (df['Age'].isnull()) & (df['Pclass'] == 2) & (df['IsAlone'] == 1), 'Age'] = 34

    # Fill all nans in the Embarked column.
    df['Embarked'] = df['Embarked'].fillna('S')

    # Get dummies for Sex and Embarked columns.
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

    # Drop unused columns.
    drop_elements = ['Name', 'Cabin', 'SibSp', 'Ticket']
    df.drop(drop_elements, axis=1, inplace=True)

    # Fill rem. nan val. 0. Per François Chollet, TF will learn to ignore 0s if they are not feature base value.
    df.fillna(value=0, inplace=True)

    return df
