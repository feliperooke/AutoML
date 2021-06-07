from .BaseWrapper import BaseWrapper
from fbprophet import Prophet
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import logging


class ProphetWrapper(BaseWrapper):
    def __init__(self, automl_instance):
        super().__init__(automl_instance)
        
        # passing info to warning level
        logging.getLogger('fbprophet').setLevel(logging.WARNING)

    def transform_data(self, data):
        self.data = data

        self.index_label = self.automl.index_label
        self.target_label = self.automl.target_label

        self.data.rename(columns={
            self.index_label: 'ds',
            self.target_label: 'y'
        }, inplace=True)

        # detecting the time frequency
        time_diffs = self.data['ds'][1:].values - self.data['ds'][:-1].values
        unique, counts = np.unique(time_diffs, return_counts=True)
        time_counts = dict(zip(unique, counts))

        # select the frequency with more occurences
        self.time_freq = max(time_counts, key=time_counts.get)

        train_size = int(len(self.data) * self.automl.train_val_split)

        self.training = self.data.iloc[:train_size]
        self.validation = self.data.iloc[train_size:]
        self.last_x = self.validation.iloc[[-1]]

    def train(self, model_params):
        # uncertainty_samples = False to speed up the prediction
        self.model = Prophet(uncertainty_samples=False, **model_params)

        self.model.fit(self.training)

        self.qmodels = [Prophet(interval_width=quantile, **model_params)
                        for quantile in self.quantiles]

        for qmodel in self.qmodels:
            qmodel.fit(self.training)

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

        date_column = self.index_label if self.index_label in X.columns else 'ds'

        def step_prediction(row, future_steps):
            future_dates = pd.date_range(
                start=row[date_column],
                periods=future_steps,
                freq=pd.Timedelta(self.time_freq))

            future_df = pd.DataFrame({'ds': future_dates})

            if quantile:
                prediction = [qmodel.predict(future_df).yhat_upper
                              for qmodel in self.qmodels]
                prediction = np.array(prediction)
                prediction = prediction.T
            else:
                prediction = self.model.predict(future_df).yhat

            return prediction

        Y_hat = X.apply(lambda row: step_prediction(row, future_steps), axis=1)

        Y_hat = Y_hat.to_numpy()

        return Y_hat

    def auto_ml_predict(self, X, future_steps, quantile, history):
        # date column to datetime type
        X[self.index_label] = pd.to_datetime(X[self.index_label])
        # removing timezone
        X[self.index_label] = X[self.index_label].dt.tz_localize(None)

        X.rename(columns={
            self.index_label: 'ds',
            self.target_label: 'y'
        }, inplace=True)

        y = self.predict(X, future_steps, quantile=quantile)
        return y

    def next(self, X, future_steps, quantile):
        return self.predict(self.last_x, future_steps, quantile=quantile)

    params_list = [{
        'growth': 'linear',
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': 'auto',
        'seasonality_mode': 'additive',
    },{
        'growth': 'linear',
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': 'auto',
        'seasonality_mode': 'multiplicative',
    }]

    @staticmethod
    def _evaluate(auto_ml, cur_wrapper):
        prefix = 'Prophet'

        print(f'Evaluating {prefix}')

        wrapper_list = []

        y_val_matrix = auto_ml._create_validation_matrix(
            val_y=cur_wrapper.validation['y'].values.T)

        for c, params in tqdm(enumerate(ProphetWrapper.params_list)):
            cur_wrapper.train(params)

            auto_ml.evaluation_results[prefix+str(c)] = {}

            y_pred = cur_wrapper.predict(
                cur_wrapper.validation, max(auto_ml.important_future_timesteps))
            
            # selecting only the important timesteps
            y_pred = y_pred[:, [-(n-1) for n in auto_ml.important_future_timesteps]]
            y_pred = y_pred[:-max(auto_ml.important_future_timesteps), :]

            auto_ml.evaluation_results[prefix +
                                    str(c)]['default'] = auto_ml._evaluate_model(y_val_matrix.T, y_pred)

            # quantile values
            q_pred = cur_wrapper.predict(
                cur_wrapper.validation, max(auto_ml.important_future_timesteps), quantile=True)

            # selecting only the important timesteps
            for i, pred in enumerate(q_pred):
                q_pred[i] = pred[[-(n-1) for n in auto_ml.important_future_timesteps], :]

            for i in range(len(auto_ml.quantiles)):
                quantile = auto_ml.quantiles[i]
                qi_pred = [pred[:, i] for pred in q_pred]
                qi_pred = qi_pred[:-max(auto_ml.important_future_timesteps)]

                auto_ml.evaluation_results[prefix + str(c)][str(
                    quantile)] = auto_ml._evaluate_model(y_val_matrix.T, qi_pred, quantile)

            wrapper_list.append(copy.copy(cur_wrapper))

        return prefix, wrapper_list
