from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from models import build_keras_model, build_xgboost_model, compile_keras_model, fit_keras_model, fit_xgboost_model
from models import keras_predict, xgboost_predict
from utils import average_results
import numpy as np

# Cross-validation params.
KFOLD_PARAMS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 19
}


def k_fold_test(x, y):
    # Split our training dataset into train/test datasets.
    kfold = StratifiedKFold(**KFOLD_PARAMS)

    # List of cross-validation scores.
    cv_scores = []

    # Cross-validate on our models.
    for train, test in kfold.split(x, y):
        model_keras = build_keras_model(x.shape[1])

        # Compile keras model.
        model_keras = compile_keras_model(model_keras)

        # Fit Keras model.
        model_keras = fit_keras_model(model_keras, x[train], y[train])

        # Predict on keras model.
        keras_predictions = keras_predict(model_keras, x_predict=x[test])

        # Build xgboost model.
        model_xgboost = build_xgboost_model()

        # Fit xgboost model.
        model_xgboost = fit_xgboost_model(model_xgboost, x[train], y[train])

        # Predict on xgboost model.
        xgboost_predictions = xgboost_predict(model_xgboost, x[test])

        # Average model results.
        results = average_results(keras_predictions, xgboost_predictions)

        # Generate kaggle csv.
        predictions = np.round(results).astype('int')

        # Evaluate the models.
        scores = accuracy_score(y[test], predictions)
        combined_scores = (scores * 100)
        print(" %.2f%%" % combined_scores)
        cv_scores.append(combined_scores)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))
