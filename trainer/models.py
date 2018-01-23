from __future__ import print_function
from trainer.utils import predictions_to_csv
from keras.models import Sequential
from trainer.model_helpers import KerasHelper
from sklearn.model_selection import train_test_split
from datetime import datetime
from keras.optimizers import RMSprop
import numpy as np


# set the logging path for ML Engine logging to Storage bucket.
def _set_cloud_log_path(job_dir):
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))


# Passenger probability of survival is predicted on keras model.
def _predict_proba(model, x_pred, batch_size):
    preds = np.round(model.predict_proba(x_pred, batch_size=batch_size)).astype(int)
    preds = preds.reshape(-1, 1).ravel()
    return preds


def train_cloud(x, y, x_pred, passenger_ids, cloud_train, train_file='gs://keras-titanic-models/data/',
                job_dir='gs://keras-titanic-models/data/',
                dropout_one=0.2, epochs_one=100, units_one=8, rms_one=0.001, batchsize_one=1, **args):
    _set_cloud_log_path(job_dir=job_dir)

    # Get NN feature input shape.
    x_shape = x.shape[1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=19)

    # Build and compile the sequential model.
    k1_model = KerasHelper(model=Sequential).build_model(input_dim=x_shape,
                                                         units=units_one,
                                                         dropout=dropout_one,
                                                         optimizer=RMSprop(rms_one))

    # Fit each model with the best params from GCMLE.
    k1_model.fit(x=x_train,
                 y=y_train,
                 epochs=epochs_one,
                 validation_data=(x_test, y_test),
                 verbose=0,
                 batch_size=batchsize_one,
                 **args)

    # Evaluate the accuracy of our model on unseen data.
    score = k1_model.evaluate(x=x_test,
                              y=y_test,
                              batch_size=batchsize_one,
                              verbose=0)

    # Print model test loss and accuracy score.
    print('Model 1:\n Test Loss: %s\n Test Accuracy: %s\n' % (score[0], score[1]))

    # Get train/predict probabilities of each model.
    pred = _predict_proba(model=k1_model,
                          x_pred=x_pred,
                          batch_size=batchsize_one)

    # Generate kaggle submission file.
    predictions_to_csv(predictions=pred, passenger_ids=passenger_ids, cloud_train=cloud_train)
