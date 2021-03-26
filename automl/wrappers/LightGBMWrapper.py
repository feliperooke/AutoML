from .BaseWrapper import BaseWrapper
from sklearn.model_selection import train_test_split


class LightGBMWrapper(BaseWrapper):
    def __init__(self, quantiles):
        super.__init__(quantiles)

    def transform_data(self, data, past_lags, index_label, target_label):
        self.past_lags = past_lags
        self.oldest_lag = int(max(self.past_lags)) + 1
        self.index_label = index_label
        self.target_label = target_label

        X = data[past_lags]
        y = data[target_label]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)

        self.training = (X_train, y_train)
        self.validation = (X_test, y_test)

    def train(self):
        pass

    def predict(self, X, future_steps, quantile=False):
        pass

    def next(self, future_steps, quantile=False):
        pass
