from .BaseWrapper import BaseWrapper
import statsmodels.api as sm
from tqdm import tqdm
import numpy as np
from numpy.linalg import LinAlgError
import itertools
import random
import copy
from sklearn.model_selection import train_test_split


class SarimaWrapper(BaseWrapper):
    def __init__(self, automl_instance):
        super().__init__(automl_instance)
        random.seed(42)

    def transform_data(self, data):
        self.automl._data_shift.transform(data)
        self.data = data
        self.past_labels = self.automl._data_shift.past_labels
        self.past_lags = self.automl._data_shift.past_lags

        # the seasonal part is determinated by the last lag
        self.seasonality = int(max(self.past_lags))
        self.oldest_lag = self.seasonality + 1

        self.index_label = self.automl.index_label
        self.target_label = self.automl.target_label

        train_size = int(len(self.data) * self.automl.train_val_split)

        self.training = self.data.iloc[:train_size]
        self.validation = self.data.iloc[train_size:]

        self.last_x = self.validation[self.target_label]
        
        # Setting parameters possibilities
        # set parameter range
        number_of_possibilities = min([len(self.past_lags), 5])
        
        lag_possibilities = list(self.past_lags)
        lag_possibilities.remove(self.seasonality) # removing the lag duplication
        p = random.sample(lag_possibilities, number_of_possibilities-1)
        d = random.sample(list(self.past_lags), number_of_possibilities)
        q = random.sample(lag_possibilities, number_of_possibilities-1)

        # list of all parameter combos
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = list(itertools.product(p, d, q, [self.seasonality]))
        combinations = list(itertools.product(pdq, seasonal_pdq))

        self.params_list = list(map(lambda params: {'order': params[0], 'seasonal_order': params[1]}, combinations))

    def train(self, model_params, quantile_params=None):
        try:
            to_train = self.training.set_index(self.index_label)
            pre_model = sm.tsa.statespace.SARIMAX(to_train, **model_params)
            self.model = pre_model.fit(maxiter=35)
            del to_train
        except LinAlgError as er: # can occur an matrix generation error during the calculations
            print(er)

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

        Y_hat = np.zeros((len(X), future_steps, len(self.quantiles))
                         ) if quantile else np.zeros((len(X), future_steps))

        if quantile:
            # not implemented
            pass
        else:
            for i in range(len(X)):
                local_model = copy.copy(self.model)
                local_model.apply(X[self.target_label][:i])
                cur_y_hat = local_model.forecast(steps=future_steps)
                Y_hat[i] = cur_y_hat
                del local_model

        return Y_hat

    def auto_ml_predict(self, X, future_steps, quantile, history):
        y = self.predict(X, future_steps, quantile=quantile)
        return y

    def next(self, X, future_steps, quantile):
        local_model = copy.copy(self.model)
        local_model.apply(self.last_x)

        prediction = local_model.forecast(steps=future_steps)

        return prediction.to_numpy()

    @staticmethod
    def _evaluate(auto_ml, cur_wrapper):
        prefix = 'SARIMA'

        print(f'Evaluating {prefix}')

        wrapper_list = []

        random_params = random.sample(cur_wrapper.params_list, 5)

        y_val_matrix = auto_ml._create_validation_matrix(
            val_y=cur_wrapper.validation[cur_wrapper.target_label].values.T)

        for c, params in tqdm(enumerate(random_params)):

            cur_wrapper.train(params)

            auto_ml.evaluation_results[prefix+str(c)] = {}

            y_pred = np.array(cur_wrapper.predict(
                cur_wrapper.validation, max(auto_ml.important_future_timesteps)))[:, [-(n-1) for n in auto_ml.important_future_timesteps]]

            y_pred = y_pred[:-max(auto_ml.important_future_timesteps), :]
            auto_ml.evaluation_results[prefix +
                                    str(c)]['default'] = auto_ml._evaluate_model(y_val_matrix.T, y_pred)

            # quantile values
            # q_pred = np.array(cur_wrapper.predict(
            #     cur_wrapper.validation[0], max(auto_ml.important_future_timesteps), quantile=True))[:, [-(n-1) for n in auto_ml.important_future_timesteps], :]

            wrapper_list.append(copy.copy(cur_wrapper))

        return prefix, wrapper_list
