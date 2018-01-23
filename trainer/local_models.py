from __future__ import print_function
from trainer.utils import predictions_to_csv
from keras.models import Sequential
from trainer.model_helpers import KerasHelper
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from xgboost import XGBClassifier
import numpy as np

# Show log output during training and evaluation.
_VERBOSE = 0

# Keras 1 hidden layer params.
_PARAMS_K1 = {
    'batch_size': 32,
    'epochs': 149,
    'units': 23,
    'dropout': 0.16491542915361779,
    'optimizer': RMSprop(lr=0.0045913049910435155)
}

# Keras 1 hidden layer params.
_PARAMS_K2 = {
    'batch_size': 32,
    'epochs': 100,
    'units': 19,
    'dropout': 0.376226444419632719,
    'optimizer': RMSprop(lr=0.0039106062204943568)
}

# Keras 1 hidden layer params.
_PARAMS_K3 = {
    'batch_size': 60,
    'epochs': 72,
    'units': 29,
    'dropout': 0.35854006364722213,
    'optimizer': RMSprop(lr=0.0097655287637879243)
}

# STACKED XGBoost params
_PARAMS_XGBOOST = {
    'max_depth': 4,
    'random_state': 19,
    'learning_rate': 0.1,
    'n_estimators': 5000,
    'min_child_weight': 1,
    'gamma': 0.9,
    'objective': 'binary:logistic',
    'scale_pos_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}


# Stacked xgboost model where passenger survivals are predicted. XGBoost model takes in all 3 keras models.
def _stacked_xgboost(x_train, y_train, x_pred):
    clf = XGBClassifier(**_PARAMS_XGBOOST).fit(x_train, y_train)
    preds = clf.predict(x_pred)
    return preds


# Passenger probability of survival is predicted on each keras model.
def _predict_proba(model, x_train, x_pred, batch_size):
    preds = np.round(model.predict_proba(x_pred, batch_size=batch_size)).astype(int)
    preds_train = np.round(model.predict_proba(x_train, batch_size=batch_size)).astype(int)
    preds = preds.reshape(-1, 1)
    preds_train = preds_train.reshape(-1, 1)
    return preds, preds_train


def train_local(x, y, x_pred, passenger_ids, cloud_train):
    # Get NN feature input shape.
    x_shape = x.shape[1]

    # Create train/test splits.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=19)

    # Build and compile the 3 best sequential models.
    k1_model = KerasHelper(model=Sequential).build_model(input_dim=x_shape,
                                                         units=_PARAMS_K1.get('units'),
                                                         dropout=_PARAMS_K1.get('dropout'),
                                                         optimizer=_PARAMS_K1.get('optimizer'))

    k2_model = KerasHelper(model=Sequential).build_model(input_dim=x_shape,
                                                         units=_PARAMS_K2.get('units'),
                                                         dropout=_PARAMS_K2.get('dropout'),
                                                         optimizer=_PARAMS_K2.get('optimizer'))

    k3_model = KerasHelper(model=Sequential).build_model(input_dim=x_shape,
                                                         units=_PARAMS_K3.get('units'),
                                                         dropout=_PARAMS_K3.get('dropout'),
                                                         optimizer=_PARAMS_K3.get('optimizer'))

    # Fit each model with the best params from GCMLE.
    k1_model.fit(x=x_train,
                 y=y_train,
                 epochs=_PARAMS_K1.get('epochs'),
                 validation_data=(x_test, y_test),
                 verbose=_PARAMS_K1.get('verbose'),
                 batch_size=_PARAMS_K1.get('batch_size'))

    k2_model.fit(x=x_train,
                 y=y_train,
                 epochs=_PARAMS_K2.get('epochs'),
                 validation_data=(x_test, y_test),
                 verbose=_PARAMS_K2.get('verbose'),
                 batch_size=_PARAMS_K2.get('batch_size'))

    k3_model.fit(x=x_train,
                 y=y_train,
                 epochs=_PARAMS_K3.get('epochs'),
                 validation_data=(x_test, y_test),
                 verbose=_PARAMS_K3.get('verbose'),
                 batch_size=_PARAMS_K3.get('batch_size'))

    # Evaluate the accuracy of our models on unseen data.
    score1 = k1_model.evaluate(x=x_test,
                               y=y_test,
                               batch_size=_PARAMS_K1.get('batch_size'),
                               verbose=_VERBOSE)

    score2 = k2_model.evaluate(x=x_test,
                               y=y_test,
                               batch_size=_PARAMS_K2.get('batch_size'),
                               verbose=_VERBOSE)

    score3 = k3_model.evaluate(x=x_test,
                               y=y_test,
                               batch_size=_PARAMS_K3.get('batch_size'),
                               verbose=_VERBOSE)

    # Print test loss and accuracy scores for each model.
    print('Model 1:\n Test Loss: %s\n Test Accuracy: %s\n' % (score1[0], score1[1]))
    print('Model 2:\n Test Loss: %s\n Test Accuracy: %s\n' % (score2[0], score2[1]))
    print('Model 3:\n Test Loss: %s\n Test Accuracy: %s\n' % (score3[0], score3[1]))

    # Get train/predict probabilities of each model.
    pred1, pred1_train = _predict_proba(model=k1_model,
                                        x_train=x_train,
                                        x_pred=x_pred,
                                        batch_size=_PARAMS_K1.get('batch_size'))

    pred2, pred2_train = _predict_proba(model=k2_model,
                                        x_train=x_train,
                                        x_pred=x_pred,
                                        batch_size=_PARAMS_K2.get('batch_size'))

    pred3, pred3_train = _predict_proba(model=k3_model,
                                        x_train=x_train,
                                        x_pred=x_pred,
                                        batch_size=_PARAMS_K3.get('batch_size'))

    # Concat train and test scores. These will be fed into the stacked xgboost model for final predictions.
    x_train_scores = np.concatenate((pred1_train, pred2_train, pred3_train), axis=1)
    x_pred_scores = np.concatenate((pred1, pred2, pred3), axis=1)

    # Get final predictions from stacked xgboost model.
    passenger_predictions = _stacked_xgboost(x_train=x_train_scores, y_train=y_train, x_pred=x_pred_scores)

    # Generate kaggle submission file.
    predictions_to_csv(predictions=passenger_predictions, passenger_ids=passenger_ids, cloud_train=cloud_train)
