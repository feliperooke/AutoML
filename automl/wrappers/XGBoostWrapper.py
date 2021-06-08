from .BaseWrapper import BaseWrapper
from tqdm import tqdm
from functools import partial
import copy
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split

# Code by: https://towardsdatascience.com/regression-prediction-intervals-with-xgboost-428e0a018b


class XGBQuantile(XGBRegressor):
    def __init__(self, quant_alpha=0.95, quant_delta=1.0, quant_thres=1.0, quant_var=1.0, base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                 n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1):
        self.quant_alpha = quant_alpha
        self.quant_delta = quant_delta
        self.quant_thres = quant_thres
        self.quant_var = quant_var

        super().__init__(base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel,
                         colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate, max_delta_step=max_delta_step,
                         max_depth=max_depth, min_child_weight=min_child_weight, missing=missing, n_estimators=n_estimators,
                         n_jobs=n_jobs, nthread=nthread, objective=objective, random_state=random_state,
                         reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, seed=seed,
                         silent=silent, subsample=subsample)

        self.test = None

    def fit(self, X, y):
        super().set_params(objective=partial(XGBQuantile.quantile_loss, alpha=self.quant_alpha,
                                             delta=self.quant_delta, threshold=self.quant_thres, var=self.quant_var))
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        y_pred = super().predict(X)
        score = XGBQuantile.quantile_score(y, y_pred, self.quant_alpha)
        score = 1./score
        return score

    @staticmethod
    def quantile_loss(y_true, y_pred, alpha, delta, threshold, var):
        x = y_true - y_pred
        grad = (x < (alpha-1.0)*delta)*(1.0-alpha) - ((x >= (alpha-1.0)
                                                       * delta) & (x < alpha*delta))*x/delta-alpha*(x > alpha*delta)
        hess = ((x >= (alpha-1.0)*delta) & (x < alpha*delta))/delta

        grad = (np.abs(x) < threshold)*grad - (np.abs(x) >= threshold) * \
            (2*np.random.randint(2, size=len(y_true)) - 1.0)*var
        hess = (np.abs(x) < threshold)*hess + (np.abs(x) >= threshold)
        return grad, hess

    @staticmethod
    def original_quantile_loss(y_true, y_pred, alpha, delta):
        x = y_true - y_pred
        grad = (x < (alpha-1.0)*delta)*(1.0-alpha)-((x >= (alpha-1.0)
                                                     * delta) & (x < alpha*delta))*x/delta-alpha*(x > alpha*delta)
        hess = ((x >= (alpha-1.0)*delta) & (x < alpha*delta))/delta
        return grad, hess

    @staticmethod
    def quantile_score(y_true, y_pred, alpha):
        score = XGBQuantile.quantile_cost(x=y_true-y_pred, alpha=alpha)
        score = np.sum(score)
        return score

    @staticmethod
    def quantile_cost(x, alpha):
        return (alpha-1.0)*x*(x < 0)+alpha*x*(x >= 0)

    @staticmethod
    def get_split_gain(gradient, hessian, l=1):
        split_gain = list()
        for i in range(gradient.shape[0]):
            split_gain.append(np.sum(gradient[:i])/(np.sum(hessian[:i])+l)+np.sum(
                gradient[i:])/(np.sum(hessian[i:])+l)-np.sum(gradient)/(np.sum(hessian)+l))

        return np.array(split_gain)


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

        self.qmodels = [XGBQuantile(quant_alpha=quantil, **model_params)
                        for quantil in self.quantiles]

        for qmodel in self.qmodels:
            qmodel.fit(self.training[0], self.training[1])

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
            q_pred = np.array(cur_wrapper.predict(
                cur_wrapper.validation[0], max(auto_ml.important_future_timesteps), quantile=True))[:, [-(n-1) for n in auto_ml.important_future_timesteps], :]

            for i in range(len(auto_ml.quantiles)):
                quantile = auto_ml.quantiles[i]
                qi_pred = q_pred[:, :, i]
                qi_pred = qi_pred[:-max(auto_ml.important_future_timesteps), :]

                auto_ml.evaluation_results[prefix + str(c)][str(
                    quantile)] = auto_ml._evaluate_model(y_val_matrix.T, qi_pred, quantile)

            wrapper_list.append(copy.copy(cur_wrapper))

        return prefix, wrapper_list
