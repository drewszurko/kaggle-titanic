from __future__ import print_function
from models import build_keras_model, compile_keras_model, fit_keras_model, keras_predict
from models import build_xgboost_model, fit_xgboost_model, xgboost_predict
from model_cv import k_fold_test
from utils import average_results
from utils import predictions_to_csv
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import data_utils

np.set_printoptions(threshold=np.inf)

TEST_MODEL = False

if __name__ == '__main__':
    # Load training and prediction dataframes.
    df_train, df_predict = data_utils.get_dataset()

    # Split features and targets.
    X = df_train.iloc[:, 1:].values
    y = df_train.iloc[:, 0].values

    # Load predict feature values.
    X_predict = df_predict.iloc[:, 0:].values

    # Scale our data.
    scaler = MinMaxScaler()
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    X_predict = scaler.transform(X_predict)

    # Predict or cross-validate model.
    if not TEST_MODEL:
        # Build keras model.
        model_keras = build_keras_model(X.shape[1])

        # Compile keras model.
        model_keras = compile_keras_model(model_keras)

        # Fit keras model.
        model_keras = fit_keras_model(model_keras, X, y)

        # Predict on keras model.
        keras_predictions = keras_predict(model_keras, x_predict=X_predict)

        # Build xgboost model.
        model_xgboost = build_xgboost_model()

        # Fit xgboost model.
        model_xgboost = fit_xgboost_model(model_xgboost, X, y)

        # Predict on xgboost model.
        xgboost_predictions = xgboost_predict(model_xgboost, X_predict)

        # Average keras and xgboost model predictions.
        results = average_results(keras_predictions, xgboost_predictions)

        # Generate kaggle predictions csv.
        predictions = np.round(results).astype('int')
        predictions_to_csv(results, df_predict)
    else:
        k_fold_test(X, y)
