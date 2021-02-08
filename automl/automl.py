import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.metrics import mean_squared_error
from .metrics import weighted_quantile_loss, weighted_absolute_percentage_error
from .transformer import DataShift


class AutoML:
    def __init__(self, path, jobs=-1, fillnan='ffill', max_input_size=40, nlags=24):
        """
        AutoML is an auto machine learning project with focus on predict
        time series using simple usage and high-level understanding over
        time series prediction methods.

        :param path: 
            Path to the input csv. The csv must be in the format (date, value).

        :param jobs:
            Number of jobs used during the training and evaluation.

        :param fillnan: {'bfill', 'ffill'}, default ffill
            Method to use for filling holes. 
            ffill: propagate last valid observation forward to next valid.
            bfill: use next valid observation to fill gap.

        """

        warnings.filterwarnings('ignore')

        self.path = path
        self.jobs = jobs
        self.fillnan = fillnan
        self.nlags = nlags

        self.data = pd.read_csv(self.path)
        self.input_transformation = lambda x: x
        self.oldest_lag = 1
        self.quantiles = [.1, .5, .9]

        if len(self.data.columns) > 2:
            raise Exception('Data has more than 2 columns.')

        self._data_shift = DataShift(nlags=self.nlags)
        self.X, self.y = self._transform_data('lightgbm')

        # results obtained during evaluation
        self.evaluation_results = {}

        # the chosen model
        self.model = None
        self.quantile_models = []

        self._evaluate()

        # train model
        self._trainer()

    def _transform_data(self, model):
        """
        Clean and prepare data to use as model input.

        :param model: Adapt data to the specific model.

        """
        data = self.data.copy()

        index_label, target_label = tuple(data.columns)

        # date column to datetime type
        data[index_label] = pd.to_datetime(data[index_label])

        # find the best past lags value
        self._data_shift.fit(data)
        self.oldest_lag = max(self._data_shift.past_lags) + 1

        # add to the data the past lags
        data = self._data_shift.transform(data)
        past_labels = self._data_shift.past_labels

        x = None
        y = None
        x_labels = past_labels  # incluir index_labels em alguns modelos

        # adapt the data to the chosen model
        if model == 'lightgbm':
            x = data[x_labels]
            y = data[target_label]

        return x, y

    def _evaluate_model(self, model, X_val, y_val, quantile=None):
        """
        Evaluate a specifc model given the data to be tested.

        :param model: Model to be evaluated.
        :param X_val: X input to generate the predictions.
        :param y_val: y values that represents the real values.
        :param quantile: Quantile value that will be evaluated.

        """

        y_pred = model.predict(X_val)

        results = {}

        if quantile is not None:
            results['wql'] = weighted_quantile_loss(quantile, y_val, y_pred)
        else:
            results['wape'] = weighted_absolute_percentage_error(y_val, y_pred)
            results['rmse'] = mean_squared_error(y_val, y_pred, squared=False)

        return results

    def _evaluate(self):
        """
        Compare baseline models

        """
        # Vamos fazer com os modelos sempre usando a api do Scikit Learn pq a gnt vai usar ele para o RandomSearch

        # using quantile prediction as default
        quantile_params = {
            'objective': 'quantile',
            'metric': 'quantile',
        }
        quantile_models = [lgb.LGBMRegressor(alpha=quantil, **quantile_params)
                           for quantil in self.quantiles]

        best_model = lgb.LGBMRegressor()  # Temp

        self.model = best_model
        self.quantile_models = quantile_models

    def _trainer(self):
        """
        Train the chosen model and evaluate the final result.

        """

        # train data shifted by the max lag period
        X_train, y_train = self.X[:-self.oldest_lag], self.y[:-self.oldest_lag]

        self.model.fit(X_train, y_train)

        for quantile_model in self.quantile_models:
            quantile_model.fit(X_train, y_train)

        # evaluate the models on the last max lag period
        X_val, y_val = self.X[-self.oldest_lag:], self.y[-self.oldest_lag:]

        # default model
        self.evaluation_results['default'] = self._evaluate_model(
            self.model, X_val, y_val)

        # quantile models
        for quantile, model in zip(self.quantiles, self.quantile_models):
            self.evaluation_results[str(quantile)] = self._evaluate_model(
                model, X_val, y_val, quantile)

    def predict(self, X, future_steps, quantile=False):
        """
        Uses the input "X" to predict "future_steps" steps into the future.

        :param X:
            Values to make a prediction with.

        :param future_steps:
            Number of steps in the future to predict.

        :param quantile:
            Use quantile models instead of the mean based.

        """
        if(len(X) < self.oldest_lag):
            raise Exception(f'''Error, to make a prediction X needs to be at
                                least {self.oldest_lag} items long''')

        cur_X = self._data_shift.transform(X.copy())
        cur_X = cur_X[self._data_shift.past_labels].values
        y = []

        for i in range(future_steps):

            if quantile:  # predict with quantile models
                predict = []
                for quantile_model in self.quantile_models:
                    predict.append(quantile_model.predict(
                        cur_X[-1].reshape(1, -1)))
                y.append(predict)

                # choose the median prediction to feed the new predictions
                predict = predict[1]

            else:  # predict with mean model
                predict = self.model.predict(cur_X[-1].reshape(1, -1))
                y.append(predict)
            # y.append(self.model.predict(np.squeeze(self.input_transformation(cur_X[-self.oldest_lag:]))))
            new_input = np.append(cur_X[-1][1:], predict, axis=0)
            cur_X = np.append(cur_X, [new_input], axis=0)

        return y

    def next(self, future_steps, quantile=False):
        """
        Predicts the next "future_steps" steps into the future using the data inserted for training.

        :param future_steps:
            Number of steps in the future to predict.

        :param quantile:
            Use quantile models instead of the mean based.

        """
        return self.predict(self.data, future_steps, quantile)

    def add_new_data(self, new_data_path, append=True):
        """
        Retrain data with the new input. 

        Obs.: It can change the number of past lags.

        :param new_data_path:
            New data path to be added.

        :param append:
            Append new data or substitute.

        """

        new_data = pd.read_csv(new_data_path)
        if len(new_data.columns) > 2:
            raise Exception('Data has more than 2 columns.')

        # new_data = self.input_transformation(new_data)

        if append:
            self.data = self.data.append(new_data, ignore_index=True)
        else:
            self.data = new_data

        self.X, self.y = self._transform_data('lightgbm')

        # self._evaluate()
        self._trainer()
