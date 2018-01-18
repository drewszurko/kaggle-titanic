from __future__ import print_function
from tqdm import tqdm
import requests
import os
import errno
import pandas as pd
import numpy as np
import re

# Local/remote location of titanic datasets.
_PASSENGER_DATA_PATH = 'data/passengerTrain.csv'
_PASSENGER_DATA_URL = 'https://github.com/drewszurko/kaggle-titanic/blob/master/data/passengerTrain.csv'

_PREDICT_PASSENGER_DATA_PATH = 'data/passengerTest.csv'
_PREDICT_PASSENGER_DATA_URL = 'https://github.com/drewszurko/kaggle-titanic/blob/master/data/passengerTest.csv'


def get_dataset():
    print("\nImporting data... Please wait.")
    # Create 'data' directory if it does not exist.
    dir_availability()

    # Load our training data.
    data = load_dataset("Training", _PASSENGER_DATA_PATH, _PASSENGER_DATA_URL)
    predict_data = load_dataset("Predict", _PREDICT_PASSENGER_DATA_PATH, _PREDICT_PASSENGER_DATA_URL)
    return data, predict_data


# Check if 'data' directory exists. Directory must exist for app to function correctly.
def dir_availability():
    try:
        os.makedirs('data')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def created_features(df):
    # Create new feature FamilySize.
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Create new feature IsAlone from FamilySize.
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

    # Fill nan Fare values.
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Extract titles from names.
    def get_title(name):
        title_search = re.search(" ([A-Za-z]+)\.", name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    # Create a new feature Title, containing passenger names.
    df['Title'] = df['Name'].apply(get_title)

    # Group uncommon titles.
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Map Title to buckets.
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)

    # Map Fare into buckets.
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    # Age Filler.
    df['Age'] = np.where(((df['Name'].str.contains('Miss')) & (df['Age'].isnull())), 21, df['Age'])
    df['Age'] = np.where(((df['Name'].str.contains('Mrs')) & (df['Age'].isnull())), 28, df['Age'])
    df['Age'] = np.where(((df['Name'].str.contains('Master')) & (df['Age'].isnull())), 5, df['Age'])
    df['Age'] = np.where(((df['Name'].str.contains('Mr.')) & (df['Age'].isnull()) & (df['Pclass'] == 3) & (
            df['SibSp'] == 0) & (df['Parch'] == 0)), 29, df['Age'])
    df['Age'] = np.where(((df['Name'].str.contains('Mr.')) & (df['Age'].isnull()) & (df['Pclass'] == 3) & (
            ~df['SibSp'] == 0) & (~df['Parch'] == 0)), 21, df['Age'])
    df['Age'] = np.where(((df['Name'].str.contains('Mr.')) & (df['Age'].isnull()) & (df['Pclass'] == 1) & (
            df['SibSp'] == 0) & (df['Parch'] == 0)), 45, df['Age'])
    df['Age'] = np.where(((df['Name'].str.contains('Mr.')) & (df['Age'].isnull()) & (df['Pclass'] == 2) & (
            df['SibSp'] == 0) & (df['Parch'] == 0)), 34, df['Age'])

    # Map Age into buckets.
    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[df['Age'] > 64, 'Age'] = 4

    # Fill all nans in the Embarked column.
    df['Embarked'] = df['Embarked'].fillna('S')

    # Get dummies for Sex and Embarked columns.
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

    # Extra features that may help.
    df["PoorWomen"] = np.where(df.eval('Sex_female==1 and Pclass<3'), 1, 0)
    df["RichWomen"] = np.where(df.eval('Sex_female==1 and Fare>23.35'), 1, 0)
    df["PoorMale"] = np.where(df.eval('Sex_male==1 and Pclass>=2'), 1, 0)

    drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp']
    df.drop(drop_elements, axis=1, inplace=True)

    return df


# Loads our training and testing data. Try/except will redownload a dataset file if it's is corrupt or missing.
def load_dataset(name, filepath, url):
    # Specify CSV column dtypes. Reduces memory usage during import.
    dtypes = {'PassengerId': np.int32, 'Survived': np.int32, 'Pclass': np.int32, 'Name': str, 'Sex': str,
              'Age': np.float16, 'SibSp': np.int32, 'Parch': np.int32, 'Ticket': str,
              'Fare ': np.float16, 'Cabin': str, 'Embarked': str}

    try:
        df = pd.read_csv(filepath, dtype=dtypes, index_col=0)
        df = created_features(df)
        df.fillna(value=0, inplace=True)
        print("%s data imported successfully." % name)
    except Exception as e:
        print(e)
        print('\n%s data is missing or corrupt. Lets fix this!' % name)
        download_dataset(name, filepath, url)
        df = pd.read_csv(filepath, dtype=dtypes)
        print("%s data imported successfully." % name)
    return df


# Creates http request to download a missing or corrupted dataset.
def download_dataset(name, filepath, url):
    print('\nDownloading %s data.' % name)
    # Create data download request.
    r = requests.get(url, stream=True)

    # Check for successful request.
    if r.status_code != 200:
        return print("\n %s data could not be downloaded. Please try again." % name)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))

    # Writes file to 'data' directory while displaying a download progress bar in the users terminal.
    with open(filepath, 'wb') as f:
        for chunk in tqdm(iterable=r.iter_content(chunk_size=1024), total=int(total_size / 1024), unit='KB', ncols=100):
            f.write(chunk)
    print("Download complete. Importing %s data." % name)
