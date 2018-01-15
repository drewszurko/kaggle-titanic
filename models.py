from __future__ import print_function
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.layers.merge import average
from keras.optimizers import RMSprop
import xgboost as xgb
import tensorflow as tf

# Params for keras model.
PARAMS_KERAS = {
    'dropout': 0.50,
    'optimizer': RMSprop(lr=0.001),
    'units': [64],
    'epochs': 400,
    'batch_size': 256,
    'gpus': 0
}

# Params for xgboost model.
PARAMS_XGBOOST = {
    'learning_rate': 0.1,
    'max_depth': 20,
    'min_child_weight': 3,
    'n_estimators': 500
}


def build_keras_model(shape):
    # If gpus count >=2, return the multi_gpu_model.
    # 9+ gpus is not supported so Keras will throw an error prompting the user to enter valid GPU #.
    if PARAMS_KERAS.get('gpus') >= 2:
        with tf.device('/cpu:0'):
            # Create the model.
            model_keras = _create_keras_model(shape)
            # Wrap model in Keras multi_gpu_model for multi GPU support.
            return multi_gpu_model(model_keras)
    else:
        # Other return regular model b/c it supports gpu counts <=1.
        return _create_keras_model(shape)


def _create_keras_model(shape):
    inputs = Input(shape=(shape,))

    # First layer instance, returns a tensor.
    l1 = Dense(PARAMS_KERAS['units'][0], activation='relu')(inputs)
    l1 = BatchNormalization()(l1)
    l1 = Dropout(PARAMS_KERAS.get('dropout'))(l1)
    l1 = Dense(PARAMS_KERAS['units'][0], activation='relu')(l1)
    l1 = BatchNormalization()(l1)
    l1 = Dropout(PARAMS_KERAS.get('dropout'))(l1)
    l1_output = Dense(1, activation='sigmoid')(l1)

    # Second layer instance, returns a tensor.
    l2 = Dense(PARAMS_KERAS['units'][0], activation='relu')(inputs)
    l2 = BatchNormalization()(l2)
    l2 = Dropout(PARAMS_KERAS.get('dropout'))(l2)
    l2 = Dense(PARAMS_KERAS['units'][0], activation='relu')(l2)
    l2 = BatchNormalization()(l2)
    l2 = Dropout(PARAMS_KERAS.get('dropout'))(l2)
    l2 = Dense(PARAMS_KERAS['units'][0], activation='relu')(l2)
    l2 = BatchNormalization()(l2)
    l2 = Dropout(PARAMS_KERAS.get('dropout'))(l2)
    l2_output = Dense(1, activation='sigmoid')(l2)

    # Third layer instance, returns a tensor.
    l3 = Dense(PARAMS_KERAS['units'][0], activation='relu')(inputs)
    l3 = BatchNormalization()(l3)
    l3 = Dropout(PARAMS_KERAS.get('dropout'))(l3)
    l3_output = Dense(1, activation='sigmoid')(l3)

    # Average layer tensors, returns a single tensor.
    l_avg = average([l1_output, l2_output, l3_output])

    model_keras = Model(inputs, l_avg)
    return model_keras


# Compile keras model.
def compile_keras_model(model_keras):
    model_keras.compile(loss='binary_crossentropy', optimizer=PARAMS_KERAS.get('optimizer'), metrics=['accuracy'])
    return model_keras


# Fit keras model.
def fit_keras_model(model_keras, x_train, y_train):
    model_keras.fit(x_train, y_train, verbose=0, batch_size=PARAMS_KERAS.get('batch_size'),
                    epochs=PARAMS_KERAS.get('epochs'))
    return model_keras


# Predict on keras model.
def keras_predict(model_keras, x_predict):
    keras_predictions = model_keras.predict(x_predict)
    return keras_predictions


# Build xgboost model.
def build_xgboost_model():
    model_xgboost = xgb.XGBClassifier()
    return model_xgboost


# Fit xgboost model with params (defined above).
def fit_xgboost_model(model_xgboost, x_train, y_train):
    model_xgboost.set_params(**PARAMS_XGBOOST)
    model_xgboost = model_xgboost.fit(x_train, y_train)
    return model_xgboost


# Predict on xgboost.
def xgboost_predict(xgboost_model, x_predict):
    xgboost_predictions = xgboost_model.predict_proba(x_predict)
    xgboost_predictions = xgboost_predictions[:, 1:]
    return xgboost_predictions
