from __future__ import print_function
from models import fit_models, gridsearch_cv
import data_utils

if __name__ == '__main__':
    # Load training and prediction dataframes.
    df_train, df_predict = data_utils.get_dataset()

    # Split features and targets.
    X = df_train.iloc[:, 1:].values
    y = df_train.iloc[:, 0].values

    # Load predict feature values.
    X_predict = df_predict.iloc[:, 0:].values

    # Get x shape for our NN input.
    x_shape = X.shape[1]

    # Gridsearch models.
    k1_params, k2_params, xg1_params, rfc1_params, etc1_params, abc1_params, gbc1_params, svc1_params = gridsearch_cv(
        X, y, x_shape)

    # Predict on models.
    fit_models(k1_params, k2_params, xg1_params, rfc1_params, etc1_params, abc1_params, gbc1_params, svc1_params,
               X, y, X_predict, df_predict)
