from __future__ import print_function
from utils import predictions_to_csv
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from model_helpers import SklearnHelper, KerasHelper, XgboostHelper
from keras.optimizers import RMSprop
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, \
    ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np

# Cross-validation params.
_STRAT_KFOLD_PARAMS = {
    'n_splits': 5,
    'random_state': 19,
    'shuffle': True
}

# Keras 1 hidden layer params.
_PARAMS_KERAS = {
    'batch_size': [64],
    'epochs': [250],
    'units': [16],
    'dropout': [0.50],
    'optimizer': [RMSprop(lr=0.002)],
    'verbose': [0]
}

# Keras 2 hidden layer params.
_PARAMS_KERAS2 = {
    'batch_size': [64],
    'epochs': [150],
    'units': [16],
    'dropout': [0.50],
    'optimizer': [RMSprop(lr=0.002)],
    'verbose': [0]
}

# XGBoost params.
_PARAMS_XGBOOST = {
    'n_estimators': [500],
    'max_depth': [4],
    'learning_rate': [0.1],
    'min_child_weight': [1],
    'objective': ['binary:logistic'],
    'random_state': [_STRAT_KFOLD_PARAMS.get('random_state')],
    'scale_pos_weight': [1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# Random Forest params.
_PARAMS_RFC = {
    'n_estimators': [400],
    'max_depth': [4],
    'min_samples_leaf': [2],
    'max_features': ['sqrt']
}

# Extra Trees params.
_PARAMS_ETC = {
    'n_estimators': [400],
    'max_depth': [6],
    'min_samples_leaf': [2],
}

# Ada Boost params.
_PARAMS_ABC = {
    'n_estimators': [400],
    'learning_rate': [0.75],
}

# Graident Boost params.
_PARAMS_GBC = {
    'n_estimators': [400],
    'max_depth': [4],
    'min_samples_leaf': [3],
}

# Support Vector params.
_PARAMS_SVC = {
    # Do not touch probability = True, otherwise output won't work.
    'probability': [True],
    'kernel': ['rbf'],
    'C': [1]
}

# STACKED XGBoost params
_PARAMS_XGBOOST_SECOND_LEVEL = {
    'max_depth': 4,
    'random_state': _STRAT_KFOLD_PARAMS.get('random_state'),
    'learning_rate': 0.1,
    'n_estimators': 500,
    'min_child_weight': 2,
    'gamma': 0.9,
    'objective': 'binary:logistic',
    'scale_pos_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

strat_kfold = StratifiedKFold(**_STRAT_KFOLD_PARAMS)


def create_keras1(input_dim=1, units=16, dropout=0.5, optimizer='RMSprop'):
    k1 = Sequential()

    k1.add(Dense(units, input_dim=input_dim, activation='relu'))
    k1.add(Dropout(dropout))
    k1.add(Dense(units, activation='relu'))
    k1.add(Dropout(dropout))
    k1.add(Dense(1, activation='sigmoid'))

    k1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return k1


def create_keras2(input_dim=1, units=16, dropout=0.5, optimizer='RMSprop'):
    k2 = Sequential()

    k2.add(Dense(units, input_dim=input_dim, activation='relu'))
    k2.add(Dropout(dropout))
    k2.add(Dense(units, activation='relu'))
    k2.add(Dropout(dropout))
    k2.add(Dense(units, activation='relu'))
    k2.add(Dropout(dropout))
    k2.add(Dense(1, activation='sigmoid'))

    k2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return k2


def _get_train_test_scores(mdl, x, y, x_pred):
    train_scores = np.zeros((x.shape[0],))
    test_scores = np.zeros((x_pred.shape[0],))
    pred_scores = np.empty((_STRAT_KFOLD_PARAMS.get('n_splits'), x_pred.shape[0]))

    for i, (train, test) in enumerate(strat_kfold.split(x, y)):
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]

        mdl.fit(x_train, y_train)

        train_scores[test] = mdl.predict_proba(x_test)[:, 1]
        pred_scores[i, :] = mdl.predict_proba(x_pred)[:, 1]

    test_scores[:] = pred_scores.mean(axis=0)
    return train_scores.reshape(-1, 1), test_scores.reshape(-1, 1)


def gridsearch_cv(x, y, x_shape):
    _PARAMS_KERAS['input_dim'] = [x_shape]
    _PARAMS_KERAS2['input_dim'] = [x_shape]

    k1_model = KerasClassifier(build_fn=create_keras1)
    k2_model = KerasClassifier(build_fn=create_keras2)

    k1 = GridSearchCV(estimator=k1_model, verbose=0,
                      param_grid=_PARAMS_KERAS, cv=strat_kfold, scoring='accuracy',
                      n_jobs=1)
    k1_results = k1.fit(X=x, y=y)
    k1_params = k1_results.best_params_
    print('Best k1 params: %s' % k1_params)

    k2 = GridSearchCV(estimator=k2_model, verbose=0,
                      param_grid=_PARAMS_KERAS2, cv=strat_kfold, scoring='accuracy',
                      n_jobs=1)

    k2_results = k2.fit(X=x, y=y)
    k2_params = k2_results.best_params_
    print('Best k2 params: %s' % k2_params)

    xg1 = GridSearchCV(estimator=XGBClassifier(), param_grid=_PARAMS_XGBOOST, cv=strat_kfold,
                       scoring='accuracy',
                       n_jobs=1)
    xg1_results = xg1.fit(X=x, y=y)
    xg1_params = xg1_results.best_params_
    print('Best xg1 params: %s' % xg1_params)

    rfc1 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=_PARAMS_RFC, cv=strat_kfold, scoring='accuracy',
                        n_jobs=1)
    rfc1_results = rfc1.fit(X=x, y=y)
    rfc1_params = rfc1_results.best_params_
    print('Best rfc1 params: %s' % rfc1_results.best_params_)

    etc1 = GridSearchCV(estimator=ExtraTreesClassifier(), param_grid=_PARAMS_ETC, cv=strat_kfold, scoring='accuracy',
                        n_jobs=1)
    etc1_results = etc1.fit(X=x, y=y)
    etc1_params = etc1_results.best_params_
    print('Best etc1 params: %s' % etc1_results.best_params_)

    abc1 = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=_PARAMS_ABC, cv=strat_kfold, scoring='accuracy',
                        n_jobs=1)
    abc1_results = abc1.fit(X=x, y=y)
    abc1_params = abc1_results.best_params_
    print('Best abc1 params: %s' % abc1_results.best_params_)

    gbc1 = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=_PARAMS_GBC, cv=strat_kfold,
                        scoring='accuracy',
                        n_jobs=1)
    gbc1_results = gbc1.fit(X=x, y=y)
    gbc1_params = gbc1_results.best_params_
    print('Best gbc1 params: %s' % gbc1_results.best_params_)

    svc1 = GridSearchCV(estimator=SVC(), param_grid=_PARAMS_SVC, cv=strat_kfold, scoring='accuracy',
                        n_jobs=1)
    svc1_results = svc1.fit(X=x, y=y)
    svc1_params = svc1_results.best_params_
    print('Best svc1 params: %s' % svc1_results.best_params_)

    return k1_params, k2_params, xg1_params, rfc1_params, etc1_params, abc1_params, gbc1_params, svc1_params


# Fit keras model.
def fit_models(k1_params, k2_params, xg1_params, rfc1_params, etc1_params, abc1_params, gbc1_params, svc1_params, x, y,
               x_pred, df_predict):
    k1 = KerasHelper(model=KerasClassifier, build_fn=create_keras1, params=k1_params)
    k2 = KerasHelper(model=KerasClassifier, build_fn=create_keras2, params=k2_params)
    xg1 = XgboostHelper(model=XGBClassifier, seed=_STRAT_KFOLD_PARAMS.get('random_state'), params=xg1_params)
    rfc1 = SklearnHelper(model=RandomForestClassifier, seed=_STRAT_KFOLD_PARAMS.get('random_state'), params=rfc1_params)
    etc1 = SklearnHelper(model=ExtraTreesClassifier, seed=_STRAT_KFOLD_PARAMS.get('random_state'), params=etc1_params)
    abc1 = SklearnHelper(model=AdaBoostClassifier, seed=_STRAT_KFOLD_PARAMS.get('random_state'), params=abc1_params)
    gbc1 = SklearnHelper(model=GradientBoostingClassifier, seed=_STRAT_KFOLD_PARAMS.get('random_state'),
                         params=gbc1_params)
    svc1 = SklearnHelper(model=SVC, seed=_STRAT_KFOLD_PARAMS.get('random_state'), params=svc1_params)

    ktr1, kt1 = _get_train_test_scores(k1, x, y, x_pred)  # Keras Sequential 1 hidden layer
    ktr2, kt2 = _get_train_test_scores(k2, x, y, x_pred)  # Keras Sequential 2 hidden layer
    xgtr1, xgt1 = _get_train_test_scores(xg1, x, y, x_pred)  # Xgboost
    rfctr1, rfct1 = _get_train_test_scores(rfc1, x, y, x_pred)  # Random Forest
    etctr1, etct1 = _get_train_test_scores(etc1, x, y, x_pred)  # Extra Trees
    abctr1, abct1 = _get_train_test_scores(abc1, x, y, x_pred)  # Ada Boost
    gbctr1, gbct1 = _get_train_test_scores(gbc1, x, y, x_pred)  # Gradient Boosting
    svctr1, svct1 = _get_train_test_scores(svc1, x, y, x_pred)  # SVC
    x_train = np.concatenate((ktr1, ktr2, xgtr1, rfctr1, etctr1, abctr1, gbctr1, svctr1), axis=1)
    x_pred = np.concatenate((kt1, kt2, xgt1, rfct1, etct1, abct1, gbct1, svct1), axis=1)

    _stacked_xgboost(x_train, y, x_pred, df_predict)


def _stacked_xgboost(x_train, y_train, x_pred, df_predict):
    gbm = XGBClassifier(**_PARAMS_XGBOOST_SECOND_LEVEL).fit(x_train, y_train)

    predictions = gbm.predict(x_pred)

    # Generate kaggle submission file.
    predictions_to_csv(predictions, df_predict)
