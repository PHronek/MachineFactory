import pandas as pd
from sklearn import linear_model


class Application:
    def __init__(self):
        pass

    @staticmethod
    def train(file):
        data = Data(file=file)

        model = LinearRegression(data.features(), data.target())
        model.train()

        return model.coefficients()


class Data:
    def __init__(self, file):
        self.data = pd.read_csv(file, delimiter=',')

    def number_of_examples(self):
        return self.data.shape[0]

    def target(self):
        return self.data.iloc[:, 0]

    def features(self):
        return self.data.iloc[:, 1:]


class LinearRegression:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.model = None

    def train(self):
        self.model = linear_model.LinearRegression().fit(self.features, self.target)

    def coefficients(self):
        if self.model:
            return self.model.coef_
        else:
            return None
