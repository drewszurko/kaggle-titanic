from __future__ import print_function
from models import fit_models, gridsearch_cv
from sklearn.preprocessing import StandardScaler
import data_utils

if __name__ == '__main__':
    # Load training and prediction dataframes.
    df_train, df_predict = data_utils.get_dataset()

    # Split features and targets.
    X = df_train.iloc[:, 1:].values
    y = df_train.iloc[:, 0].values

    # Load predict feature values.
    X_predict = df_predict.iloc[:, 0:].values

    # Scale our data.
    scaler = StandardScaler()
    scaler = scaler.fit(X)
    X_scaled = scaler.transform(X)
    X_pred_scaled = scaler.transform(X_predict)

    x_shape = X.shape[1]

    # Gridsearch models.
    k1_params, k2_params, xg1_params, rfc1_params, etc1_params, abc1_params, gbc1_params, svc1_params = gridsearch_cv(
        X_scaled, y, x_shape)

    # Predict on models.
    fit_models(k1_params, k2_params, xg1_params, rfc1_params, etc1_params, abc1_params, gbc1_params, svc1_params,
               X_scaled, y, X_pred_scaled, df_predict)
