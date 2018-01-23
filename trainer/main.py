from __future__ import print_function
from trainer.cloud_models import train_cloud
from trainer.local_models import train_local
from sklearn.preprocessing import StandardScaler
from trainer import data_utils
import argparse


_cloud_train = False

if __name__ == '__main__':
    # Load training and prediction dataframes.
    df_train, df_predict = data_utils.get_dataset(cloud_train=_cloud_train)

    # Split features and targets values.
    X = df_train.iloc[:, 1:].values
    y = df_train.iloc[:, 0].values

    # Load predict feature values.
    X_predict = df_predict.iloc[:, 0:].values

    # Extract passenger ids for predictions output csv.
    passenger_ids = df_predict.index.values

    # Scale features to a common scale. Scale/transform prediction features to the same scale model was trained on.
    scaler = StandardScaler()
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    X_predict = scaler.transform(X_predict)

    # Is training local? If not, parse cmd line arguments for hyperparameter tuning.
    if not _cloud_train:
        # Train model locally.
        train_local(x=X, y=y, x_pred=X_predict, passenger_ids=passenger_ids, cloud_train=_cloud_train)
    else:
        cloud_parser = argparse.ArgumentParser()

        cloud_parser.add_argument(
            '--train-file',
            help='Cloud Storage bucket or local path to training data')
        cloud_parser.add_argument(
            '--job-dir',
            help='Cloud storage bucket to export the model and store temp files')
        cloud_parser.add_argument(
            '--dropout-one',
            type=float,
            help='Dropout hyperparameter')
        cloud_parser.add_argument(
            '--epochs-one',
            type=int,
            help='Model epochs')
        cloud_parser.add_argument(
            '--units-one',
            type=int,
            help='Units used for input, output, and hidden layers')
        cloud_parser.add_argument(
            '--rms-one',
            type=float,
            help='RMSProp optimizer learning rate.')
        cloud_parser.add_argument(
            '--batchsize-one',
            type=int,
            help='Batchsize for Training/predicting')

        args = cloud_parser.parse_args()
        arguments = args.__dict__

        # Train model in GCMLE.
        train_cloud(x=X, y=y, x_pred=X_predict, passenger_ids=passenger_ids, cloud_train=_cloud_train, **arguments)
