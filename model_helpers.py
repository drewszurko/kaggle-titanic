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
