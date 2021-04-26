from .BaseWrapper import BaseWrapper
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


class LightGBMWrapper(BaseWrapper):
    def __init__(self, quantiles):
        super().__init__(quantiles)

    def transform_data(self, data, past_labels, past_lags, index_label, target_label, train_val_split):
        self.data = data
        self.past_labels = past_labels
        self.past_lags = past_lags
        self.oldest_lag = int(max(self.past_lags)) + 1
        self.index_label = index_label
        self.target_label = target_label
        self.last_x = data.drop([index_label, target_label], axis=1).tail(1)

        X = data[past_labels]
        y = data[target_label]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_val_split, shuffle=False)

        self.training = (X_train, y_train)
        self.validation = (X_test, y_test)

    def train(self, model_params, quantile_params):

        self.qmodels = [lgb.LGBMRegressor(alpha=quantil, **model_params, **quantile_params)
                        for quantil in self.quantiles]

        for qmodel in self.qmodels:
            qmodel.fit(self.training[0], self.training[1])

        self.model = lgb.LGBMRegressor(**model_params)
        self.model.fit(self.training[0], self.training[1])

    def predict(self, X, future_steps, quantile=False):
        """
        Uses the input "X" to predict "future_steps" steps into the future for each os the instances in "X".

        :param X:
            Numpy array to make a prediction with, the shape of the input is (instances, steps).

        :param future_steps:
            Number of steps in the future to predict.

        :param quantile:
            Use quantile models instead of the mean based.

        """
        if(X.shape[1] < self.oldest_lag):
            raise Exception(
                f'''Error, to make a prediction X needs to have shape (n, {self.oldest_lag})''')

        Y_hat = np.zeros((len(X), future_steps, len(self.quantiles))) if quantile else np.zeros((len(X), future_steps))
        if quantile:
            for i, x in enumerate(X.values):
                cur_x = x.copy()
                for step in range(future_steps):
                    for j, qmodel in enumerate(self.qmodels):
                        cur_y_hat = qmodel.predict(
                            cur_x[self.past_lags].reshape(1,-1))
                        Y_hat[i, step, j] = cur_y_hat
                    new_x = self.model.predict(cur_x[self.past_lags].reshape(1,-1))
                    cur_x = np.roll(cur_x, -1)
                    cur_x[-1] = new_x

        else:
            for i, x in enumerate(X.values):
                cur_x = x.copy()
                for step in range(future_steps):
                    cur_y_hat = self.model.predict(cur_x[self.past_lags].reshape(1,-1))
                    Y_hat[i, step] = cur_y_hat
                    cur_x = np.roll(cur_x, -1)
                    cur_x[-1] = cur_y_hat

        return Y_hat

    def next(self, future_steps, quantile=False):
        return self.predict(self.last_x, future_steps, quantile=quantile)
