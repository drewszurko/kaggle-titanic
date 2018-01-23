from __future__ import print_function
from keras.layers import Dense, Dropout


class KerasHelper(object):
    def __init__(self, model):
        self.model = model()

    def build_model(self, input_dim, units, dropout, optimizer):
        self.model.add(Dense(units=units, input_dim=input_dim, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(units, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.model

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        x = self.model.predict(x)[:, 0]
        return x

    def predict_proba(self, x):
        return self.model.predict_proba(x)[:, 1]

    def fit(self, x, y):
        return self.model.fit(x, y)
