from lightgbm.sklearn import LGBMRegressor
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
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
    def __init__(self, path, jobs=0, fillnan='ffill', nlags=24, wrapper_constructors=[LightGBMWrapper], important_future_timesteps=[1], train_val_split=0.8):
        """
        AutoML is an auto machine learning project with focus on predict
        time series using simple usage and high-level understanding over
        time series prediction methods.

        :param path:
            Path to the input csv. The csv must be in the format (date, value).

        :param jobs:
            Number of jobs used during the training and evaluation.

        :param wrapper_constructors:
            List with the constructors of all the wrappers that will be used in this instance of AutoML.

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
        self.train_val_split = train_val_split
        self.quantiles = [.1, .5, .9]
        self.wrapper_constructors = wrapper_constructors
        self.wrappers = {wr.__name__: wr(self) for wr in wrapper_constructors}
        self.tft_wrapper = TFTWrapper(self)
        self.lightgbm_wrapper = LightGBMWrapper(self)
        self.important_future_timesteps = important_future_timesteps

        if len(self.data.columns) > 2:
            raise Exception('Data has more than 2 columns.')

        self._data_shift = DataShift(nlags=self.nlags)

        # results obtained during evaluation
        self.evaluation_results = {}

        # the chosen model
        self.model = None
        self.quantile_models = []

        self._transform_data()
        self._evaluate()

    def _transform_data(self):
        """
        Clean and prepare data to use on each wrapper.

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

        for wrapper in self.wrappers.values():
            wrapper.transform_data(data)

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

    def _create_validation_matrix(self, val_y):
        """
        Function Creating the validation matrix.
        """
        y_val = []

        for n in self.important_future_timesteps:
            y_val.append(np.roll(val_y, -(n - 1)).T)

        return np.array(y_val)[:,
                               :-(max(self.important_future_timesteps))]

    def _evaluate(self):
        """
        Compare baseline models

        """
        # Vamos fazer com os modelos sempre usando a api do Scikit Learn pq a gnt vai usar ele para o RandomSearch

        wrapper_result_dict = {}

        for cur_wrapper, cur_constructor in zip(self.wrappers.values(), self.wrapper_constructors):
            prefix, wrapper_list = cur_constructor._evaluate(self, cur_wrapper)
            wrapper_result_dict[prefix] = wrapper_list

        # Choose the best model comparing the default prediction metric results
        wape_values = {}
        for x in self.evaluation_results.items():
            wape_values[x[0]] = x[1]['default']['wape']
        min_metric = min(wape_values, key=wape_values.get)

        for prefix in sorted(list(wrapper_result_dict.keys()), key=lambda x: len(x), reverse=True):
            if prefix in min_metric:
                idx = int(min_metric[-1])
                self.model = wrapper_result_dict[prefix][idx]

    def predict(self, X, future_steps, quantile=False, history=[]):
        """
        Uses the input "X" to predict "future_steps" steps into the future.

        :param X:
            Values to make a prediction with.

        :param future_steps:
            Number of steps in the future to predict.

        :param quantile:
            Use quantile models instead of the mean based.

        :param history:
            History buffer that will be used to as base to new predictions.
            The history based models demands at least 2 * oldest_lag to make the predictions,
            so if the choosen model is one of them it demands this parameter.
            History based models: [TFT].

        """
        if(len(X) < self.oldest_lag):
            raise Exception(f'''Error, to make a prediction X needs to be at
                                least {self.oldest_lag} items long''')
        y = self.model.auto_ml_predict(X, future_steps, quantile, history)

        return y

    def next(self, future_steps, quantile=False):
        """
        Predicts the next "future_steps" steps into the future using the data inserted for training.

        :param future_steps:
            Number of steps in the future to predict.

        :param quantile:
            Use quantile models instead of the mean based.

        """
        return self.model.next(self.data, future_steps, quantile)
