import numpy as np


class SklearnHelper(object):
    def __init__(self, model, seed=0, params=None):
        params['random_state'] = seed
        self.model = model(**params)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def fit(self, x, y):
        return self.model.fit(x, y)

    def feature_importances(self, x, y):
        print(self.model.fit(x, y).feature_importances_)


class KerasHelper(object):
    def __init__(self, model, build_fn=None, params=None):
        self.model = model(build_fn, **params)

    def build_fn(self, build_fn):
        return self.model.build_fn(build_fn)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        x = self.model.predict(x)[:, 0]
        return x

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def fit(self, x, y):
        return self.model.fit(x, y)


class XgboostHelper(object):
    def __init__(self, model, seed=0, params=None):
        params['random_state'] = seed
        self.model = model(**params)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        x = np.round(self.model.predict_proba(x)[:, 1])
        return x

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def fit(self, x, y):
        return self.model.fit(x, y)
