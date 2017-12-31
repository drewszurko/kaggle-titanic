from __future__ import print_function
from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf

_CLASS_WEIGHTS = {0: 1, 1: 136}


def build_model(gpus, units, dropout):
    # If gpus count >=2, return the multi_gpu_model.
    # 9+ gpus is not supported so Keras will throw an error prompting the user to enter valid GPU #.
    if gpus >= 2:
        with tf.device('/cpu:0'):
            # Create the model.
            model = _create_model(units, dropout)
            # Wrap model in Keras multi_gpu_model for multi GPU support.
            return multi_gpu_model(model, gpus)
    else:
        # Other return regular model b/c it supports gpu counts <=1.
        return _create_model(units, dropout)


def _create_model(units, dropout):
    model = Sequential()
    model.add(Dense(units[0], input_dim=5, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units[1], activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units[2], activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(units[3], activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    return model


def compile_model(model, lr_rate):
    adam_optimizer = Adam(lr=lr_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    return model


def fit_model(model, patience, x_train, y_train):
    print("\nTraining model...\n")
    cb = EarlyStopping(monitor='val_loss', patience=patience)
    model.fit(x_train, y_train, epochs=1000, batch_size=8, callbacks=[cb], validation_split=0.1)
    return model
