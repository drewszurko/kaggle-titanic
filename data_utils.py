from __future__ import print_function
from tqdm import tqdm
import requests
import os
import errno
import pandas as pd
import numpy as np

# Local/remote location of exoplanet datasets.
_PASSENGER_DATA_PATH = 'data/passengerTrain.csv'
_PASSENGER_DATA_URL = ''


def get_dataset():
    print("\nImporting data... Please wait.")
    # Create 'data' directory if it does not exist.
    dir_availability()

    # Load our training data.
    data = load_dataset("Training", _PASSENGER_DATA_PATH, _PASSENGER_DATA_URL)

    return data


# Check if 'data' directory exists. Directory must exist for app to function correctly.
def dir_availability():
    try:
        os.makedirs('data')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


# Loads our training and testing data. Try/except will redownload a dataset file if it's is corrupt or missing.
def load_dataset(name, filepath, url):
    # Specify CSV column dtypes. Reduces memory usage during import.
    dtypes = {'PassengerId': np.long, 'Survived': np.long, 'Pclass': np.long, 'Name': str, 'Sex': str,
              'Age': np.float16, 'SibSp': np.long, 'Parch': np.long, 'Ticket': str,
              'Fare ': np.float16, 'Cabin': str, 'Embarked': str}

    try:
        df = pd.read_csv(filepath, dtype=dtypes, index_col=0)
        drop_columns = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        df.drop(drop_columns, axis=1, inplace=True)
        df.dropna(inplace=True)
        print("%s data imported successfully." % name)
    except Exception as e:
        print('\n%s data is missing or corrupt. Lets fix this!' % name)
        download_dataset(name, filepath, url)
        df = pd.read_csv(filepath, dtype=dtypes)
        print("%s data imported successfully." % name)
    # df = df.values
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
