from lightgbm.sklearn import LGBMRegressor
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.metrics import mean_squared_error
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from .metrics import weighted_quantile_loss, weighted_absolute_percentage_error
from .transformer import DataShift
from .wrappers.TFTWrapper import TFTWrapper
from .wrappers.LightGBMWrapper import LightGBMWrapper


class AutoML:
    def __init__(self, path, jobs=0, fillnan='ffill', nlags=24, important_future_timesteps=[1]):
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

        :param nlags: default 24
            Maximum number of hours in the past used for each prediction

        :param important_future_timesteps: default [1]
            Timesteps in the future that will be used in model selection
        """

        warnings.filterwarnings('ignore')

        self.path = path
        self.jobs = jobs
        self.fillnan = fillnan
        self.nlags = nlags

        self.data = pd.read_csv(self.path)
        self.target_label = None
        self.index_label = None
        self.oldest_lag = 1
        self.quantiles = [.1, .5, .9]
        self.tft_wrapper = TFTWrapper(self.quantiles)
        self.lightgbm_wrapper = LightGBMWrapper(self.quantiles)
        self.important_future_timesteps = important_future_timesteps

        if len(self.data.columns) > 2:
            raise Exception('Data has more than 2 columns.')

        self._data_shift = DataShift(nlags=self.nlags)
        self.X, self.y = self._transform_data('lightgbm')
        self.training, self.validation = self._transform_data('TFT')

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

        self.index_label, self.target_label = tuple(data.columns)

        # date column to datetime type
        data[self.index_label] = pd.to_datetime(data[self.index_label])
        # removing timezone
        data[self.index_label] = data[self.index_label].dt.tz_localize(None)

        # find the best past lags value
        self._data_shift.fit(data)
        self.oldest_lag = int(max(self._data_shift.past_lags)) + 1

        if model == 'TFT':
            # return processed_data
            self.tft_wrapper.transform_data(data,
                                            self._data_shift.past_lags,
                                            self.index_label,
                                            self.target_label)

            return (self.tft_wrapper.training, self.tft_wrapper.validation)

        # add to the data the past lags
        data = self._data_shift.transform(data)
        past_labels = self._data_shift.past_labels

        # adapt the data to the chosen model
        if model == 'lightgbm':
            self.lightgbm_wrapper.transform_data(data,
                                                 self._data_shift.past_labels,
                                                 self.index_label,
                                                 self.target_label, self.train_val_split)
            return (self.lightgbm_wrapper.training, self.lightgbm_wrapper.validation)

    def _evaluate_model(self, y_val, y_pred, quantile=None):
        """
        Evaluate a specifc model given the data to be tested.

        :param model: Model to be evaluated.
        :param X_val: X input to generate the predictions.
        :param y_val: y values that represents the real values.
        :param quantile: Quantile value that will be evaluated.

        """

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

        # creating the validation matrix
        y_val_t1 = np.array(self.y[-self.oldest_lag:])
        y_val = []

        for n in self.important_future_timesteps:
            y_val.append(np.roll(y_val_t1, -(n - 1)))

        y_val = np.array(y_val)[
            :-(max(self.important_future_timesteps) - 1)]

        # TFT

        params_list = [{
            'hidden_size': 16,
            'lstm_layers': 1,
            'dropout': 0.1,
            'attention_head_size': 1,
            'reduce_on_plateau_patience': 4,
            'hidden_continuous_size': 8,
            'learning_rate': 1e-3,
            'gradient_clip_val': 0.1,
        }, {
            'hidden_size': 32,
            'lstm_layers': 1,
            'dropout': 0.2,
            'attention_head_size': 2,
            'reduce_on_plateau_patience': 4,
            'hidden_continuous_size': 8,
            'learning_rate': 1e-2,
            'gradient_clip_val': 0.7,
        }, {
            'hidden_size': 64,
            'lstm_layers': 2,
            'dropout': 0.3,
            'attention_head_size': 3,
            'reduce_on_plateau_patience': 4,
            'hidden_continuous_size': 16,
            'learning_rate': 1e-3,
            'gradient_clip_val': 0.7,
        }, {
            'hidden_size': 64,
            'lstm_layers': 2,
            'dropout': 0.3,
            'attention_head_size': 4,
            'reduce_on_plateau_patience': 4,
            'hidden_continuous_size': 32,
            'learning_rate': 1e-2,
            'gradient_clip_val': 0.5,
        }, {
            'hidden_size': 128,
            'lstm_layers': 2,
            'dropout': 0.3,
            'attention_head_size': 4,
            'reduce_on_plateau_patience': 4,
            'hidden_continuous_size': 60,
            'learning_rate': 1e-3,
            'gradient_clip_val': 0.5,
        }, ]

        print('Evaluating TFT')

        tft_list = []
        for c, params in enumerate(params_list):
            self.tft_wrapper.train(max_epochs=50, **params)
            # evaluate the TFT
            self.evaluation_results['TFT' + str(c)] = {}

            # default values
            y_pred = np.array(self.tft_wrapper.predict(
                self.validation, max(self.important_future_timesteps)))[:, [[n-1 for n in self.important_future_timesteps]]]
            # TODO: Esse y_pred está na shape [instancia, timestamp] após a correção
            self.evaluation_results['TFT' +
                                    str(c)]['default'] = self._evaluate_model(y_val, y_pred)

            # quantile values
            q_pred = np.array(self.tft_wrapper.predict(
                self.validation, max(self.important_future_timesteps), quantile=True))[:, [[n-1 for n in self.important_future_timesteps]], :]
            # TODO: Esse y_pred está na shape [instancia, timestamp, quantile] após a correção
            tft_list.append(self.tft_wrapper.model)

            for i in range(len(self.quantiles)):
                quantile = self.quantiles[i]
                q_pred = y_pred[:, i]
                self.evaluation_results['TFT' + str(c)][str(
                    quantile)] = self._evaluate_model(y_val, q_pred, quantile)

        # LightGBM

        print('Evaluating LightGBM')

        lgbm_params_list = [{
            'num_leaves': 32,
            'max_depth': 6,
            'learning_rate': 0.001,
            'num_iterations': 15000,
            'n_estimators': 100,
        }, {
            'num_leaves': 64,
            'max_depth': 8,
            'learning_rate': 0.001,
            'num_iterations': 15000,
            'n_estimators': 200,
        }, {
            'num_leaves': 128,
            'max_depth': 10,
            'learning_rate': 0.001,
            'num_iterations': 15000,
            'n_estimators': 300,
        }, {
            'num_leaves': 128,
            'max_depth': 8,
            'learning_rate': 0.005,
            'num_iterations': 15000,
            'n_estimators': 200,
        }, {
            'num_leaves': 64,
            'max_depth': 10,
            'learning_rate': 0.001,
            'num_iterations': 15000,
            'n_estimators': 300,
        }, ]

        # using quantile prediction as default
        quantile_params = {
            'objective': 'quantile',
            'metric': 'quantile',
        }

        lgbm_list = []
        for c, params in enumerate(lgbm_params_list):
            self.evaluation_results['LightGBM'+str(c)] = {}
            self.lightgbm_wrapper.train(params, quantile_params)

            # # TODO: Adapte isso para o lightgbmwrapper
            # y_pred = np.array(self.lightgbm_wrapper.predict(
            #     self.validation, max(self.important_future_timesteps)))[:, [[n-1 for n in self.important_future_timesteps]]]
            # # TODO: Esse y_pred está na shape [instancia, timestamp] após a correção
            # self.evaluation_results['TFT' +
            #                         str(c)]['default'] = self._evaluate_model(y_val, y_pred)

            # # quantile values
            # q_pred = np.array(self.tft_wrapper.predict(
            #     self.validation, max(self.important_future_timesteps), quantile=True))[:, [[n-1 for n in self.important_future_timesteps]], :]
            # # TODO: Esse y_pred está na shape [instancia, timestamp, quantile] após a correção
            # tft_list.append(self.tft_wrapper.model)

            # for i in range(len(self.quantiles)):
            #     quantile = self.quantiles[i]
            #     q_pred = y_pred[:, i]
            #     self.evaluation_results['TFT' + str(c)][str(
            #         quantile)] = self._evaluate_model(y_val, q_pred, quantile)

        # Choose the best model comparing the default prediction metric results
        wape_values = {}
        for x in self.evaluation_results.items():
            wape_values[x[0]] = x[1]['default']['wape']
        min_metric = min(wape_values, key=wape_values.get)
        print(min_metric)
        if min_metric == 'LightGBM':
            idx = int(min_metric[-1])
            self.model = lgbm_list[idx]['default']
            self.quantile_models = lgbm_list[idx]['quantile']
        elif 'TFT' in min_metric:
            idx = int(min_metric[-1])
            self.model = tft_list[idx]

    def _trainer(self, idx=0):
        """
        Train the chosen model and evaluate the final result.

        """

        if isinstance(self.model, LGBMRegressor):
            # train data shifted by the max lag period
            X_train, y_train = self.X[:-
                                      self.oldest_lag], self.y[:-self.oldest_lag]

            self.model.fit(X_train, y_train)

            for quantile_model in self.quantile_models:
                quantile_model.fit(X_train, y_train)

            # evaluate the models on the last max lag period
            X_val, y_val = self.X[-self.oldest_lag:], self.y[-self.oldest_lag:]

            # default model
            y_pred = self.model.predict(X_val)
            self.evaluation_results['LightGBM' +
                                    str(idx)]['default'] = self._evaluate_model(y_val, y_pred)

            # quantile models
            for quantile, model in zip(self.quantiles, self.quantile_models):
                y_pred = model.predict(X_val)
                self.evaluation_results['LightGBM'+str(idx)][str(
                    quantile)] = self._evaluate_model(y_val, y_pred, quantile)

            # return self.model, self.quantile_models

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

        # Pre-process data
        if isinstance(self.model, LGBMRegressor):
            cur_X = self._data_shift.transform(X.copy())
            cur_X = cur_X[self._data_shift.past_labels].values
        y = []

        # Prediction
        if isinstance(self.model, TemporalFusionTransformer):
            y = self.tft_wrapper.predict(X, self.data, future_steps, quantile)

        else:
            for _ in range(future_steps):

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
        if isinstance(self.model, TemporalFusionTransformer):
            return self.predict(self.data, future_steps, quantile=quantile)
        else:
            return self.predict(self.data, future_steps, quantile=quantile)

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

        if append:
            self.data = self.data.append(new_data, ignore_index=True)
        else:
            self.data = new_data

        self.X, self.y = self._transform_data('lightgbm')

        # self._evaluate()
        self._trainer()
