from .BaseWrapper import BaseWrapper
from tqdm import tqdm
from functools import partial
import copy
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split

from scipy.stats import binom_test

from sklearn.base import BaseEstimator, RegressorMixin
from xgboost.sklearn import XGBRegressor
from functools import partial

# Code by: https://towardsdatascience.com/regression-prediction-intervals-with-xgboost-428e0a018b


class XGBQuantile(BaseEstimator, RegressorMixin):
    def __init__(self, quant_alpha,quant_delta,quant_thres,quant_var,
    n_estimators = 100,max_depth = 3,reg_alpha = 5.,reg_lambda=1.0,gamma=0.5):
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta 
        self.quant_thres = quant_thres 
        self.quant_var = quant_var 
        #xgboost parameters 
        self.n_estimators = n_estimators 
        self.max_depth = max_depth 
        self.reg_alpha= reg_alpha 
        self.reg_lambda = reg_lambda 
        self.gamma = gamma 
        #keep xgboost estimator in memory 
        self.clf = None 
    def fit(self, X, y): 
        def quantile_loss(y_true, y_pred,_alpha,_delta,_threshold,_var): 
            x = y_true - y_pred 
            grad = (x<(_alpha-1.0)*_delta)*(1.0-_alpha)- ((x>=(_alpha-1.0)*_delta)&
                                    (x<_alpha*_delta) )*x/_delta-_alpha*(x>_alpha*_delta) 
            hess = ((x>=(_alpha-1.0)*_delta)& (x<_alpha*_delta) )/_delta 
            _len = np.array([y_true]).size 
            var = (2*np.random.randint(2, size=_len)-1.0)*_var 
            grad = (np.abs(x)<_threshold )*grad - (np.abs(x)>=_threshold )*var 
            hess = (np.abs(x)<_threshold )*hess + (np.abs(x)>=_threshold ) 
            return grad, hess 
        self.clf = XGBRegressor(
         objective=partial( quantile_loss,
                            _alpha = self.quant_alpha,
                            _delta = self.quant_delta,
                            _threshold = self.quant_thres,
                            _var = self.quant_var), 
                            n_estimators = self.n_estimators,
                            max_depth = self.max_depth,
                            reg_alpha =self.reg_alpha, 
                            reg_lambda = self.reg_lambda,
                            gamma = self.gamma )
        self.clf.fit(X,y) 
        return self 
    def predict(self, X): 
        y_pred = self.clf.predict(X) 
        return y_pred 
    def score(self, X, y): 
        y_pred = self.clf.predict(X) 
        score = (self.quant_alpha-1.0)*(y-y_pred)*(y<y_pred)+self.quant_alpha*(y-y_pred)* (y>=y_pred) 
        score = 1./np.sum(score) 
        return score


class XGBoostWrapper(BaseWrapper):
    def __init__(self, automl_instance):
        super().__init__(automl_instance)

    def transform_data(self, data):
        self.data = self.automl._data_shift.transform(data)
        self.past_labels = self.automl._data_shift.past_labels
        self.past_lags = self.automl._data_shift.past_lags
        self.oldest_lag = int(max(self.past_lags)) + 1
        self.index_label = self.automl.index_label
        self.target_label = self.automl.target_label
        self.last_x = data.drop(
            [self.index_label, self.target_label], axis=1).tail(1)

        X = self.data[self.past_labels]
        y = self.data[self.target_label]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.automl.train_val_split, shuffle=False)

        self.training = (X_train, y_train)
        self.validation = (X_test, y_test)

    def train(self, model_params):

        # self.qmodels = [XGBQuantile(quant_delta=1,quant_thres=1, quant_var=1,quant_alpha=quantil, **model_params)
        #                 for quantil in self.quantiles]

        # for qmodel in self.qmodels:
        #     qmodel.fit(self.training[0], self.training[1])

        self.model = xgb.XGBRegressor(**model_params)
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

        Y_hat = np.zeros((len(X), future_steps, len(self.quantiles))
                         ) if quantile else np.zeros((len(X), future_steps))
        if quantile:
            for i, x in enumerate(X.values):
                cur_x = x.copy()
                for step in range(future_steps):
                    print(step,'->',cur_x)
                    for j, qmodel in enumerate(self.qmodels):
                        cur_y_hat = qmodel.predict(
                            cur_x[self.past_lags].reshape(1, -1))
                        Y_hat[i, step, j] = cur_y_hat
                    new_x = self.model.predict(
                        cur_x[self.past_lags].reshape(1, -1))
                    cur_x = np.roll(cur_x, -1)
                    cur_x[-1] = new_x

        else:
            for i, x in enumerate(X.values):
                cur_x = x.copy()
                for step in range(future_steps):
                    cur_y_hat = self.model.predict(
                        cur_x[self.past_lags].reshape(1, -1))
                    Y_hat[i, step] = cur_y_hat
                    cur_x = np.roll(cur_x, -1)
                    cur_x[-1] = cur_y_hat

        return Y_hat

    def auto_ml_predict(self, X, future_steps, quantile, history):
        X = self.automl._data_shift.transform(X)
        X = X.drop(self.index_label, axis=1)
        y = self.predict(X, future_steps, quantile=quantile)
        return y

    def next(self, X, future_steps, quantile):
        return self.predict(self.last_x, future_steps, quantile=quantile)

    # Static Values and Methods

    params_list = [{
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100,
    }, {
        'max_depth': 3,
        'learning_rate': 0.3,
        'n_estimators': 120,
    }, {
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
    }, {
        'max_depth': 4,
        'learning_rate': 0.3,
        'n_estimators': 80,
    }, {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 100,
    }, {
        'max_depth': 5,
        'learning_rate': 0.3,
        'n_estimators': 80,
    }, ]

    @staticmethod
    def _evaluate(auto_ml, cur_wrapper):
        prefix = 'XGBoost'

        print(f'Evaluating {prefix}')

        wrapper_list = []
        y_val_matrix = auto_ml._create_validation_matrix(
            cur_wrapper.validation[1].values.T)

        for c, params in tqdm(enumerate(XGBoostWrapper.params_list)):
            auto_ml.evaluation_results[prefix+str(c)] = {}
            cur_wrapper.train(params)

            y_pred = np.array(cur_wrapper.predict(
                cur_wrapper.validation[0], max(auto_ml.important_future_timesteps)))[:, [-(n-1) for n in auto_ml.important_future_timesteps]]

            y_pred = y_pred[:-max(auto_ml.important_future_timesteps), :]
            auto_ml.evaluation_results[prefix +
                                       str(c)]['default'] = auto_ml._evaluate_model(y_val_matrix.T, y_pred)

            # quantile values
            # q_pred = np.array(cur_wrapper.predict(
            #     cur_wrapper.validation[0], max(auto_ml.important_future_timesteps), quantile=True))[:, [-(n-1) for n in auto_ml.important_future_timesteps], :]

            # for i in range(len(auto_ml.quantiles)):
            #     quantile = auto_ml.quantiles[i]
            #     qi_pred = q_pred[:, :, i]
            #     qi_pred = qi_pred[:-max(auto_ml.important_future_timesteps), :]

            #     auto_ml.evaluation_results[prefix + str(c)][str(
            #         quantile)] = auto_ml._evaluate_model(y_val_matrix.T, qi_pred, quantile)

            wrapper_list.append(copy.copy(cur_wrapper))

        return prefix, wrapper_list
