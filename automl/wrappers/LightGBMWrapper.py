from .BaseWrapper import BaseWrapper


class LightGBMWrapper(BaseWrapper):
    __init__(self, quantiles)
    super.__init__(quantiles)

    def transform_data(self, data, past_lags, index_label, target_label):
        pass

    def train(self):
        pass

    def predict(self, X, future_steps, quantile=False):
        pass

    def next(self, future_steps, quantile=False):
        pass
