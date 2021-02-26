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


class AutoML:
    def __init__(self, path, jobs=0, fillnan='ffill', max_input_size=40, nlags=24):
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
        self.target_label = None
        self.index_label = None
        self.input_transformation = lambda x: x
        self.oldest_lag = 1
        self.quantiles = [.1, .5, .9]

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
            # time index are epoch values
            # data["time_idx"] = (data[self.index_label] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
            data["time_idx"] = data.index
            data['group_id'] = 'series'

            max_prediction_length = self.oldest_lag
            max_encoder_length = self.oldest_lag
            training_cutoff = data["time_idx"].max() - max_prediction_length

            training = TimeSeriesDataSet(
                data[lambda x: x.time_idx <= training_cutoff],
                time_idx="time_idx",
                group_ids=["group_id"],
                target=self.target_label,
                min_encoder_length=0,
                max_encoder_length=max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=max_prediction_length,
                static_categoricals=["group_id"],
                time_varying_unknown_reals=[self.target_label],
                # the docs says that the max_lag < max_encoder_length
                lags={self.target_label: list(self._data_shift.past_lags[1:-1] + 1)},
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,    
                # allow_missings=True
            )

            # create validation set (predict=True) which means to predict the last max_prediction_length points in time
            # for each series
            validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

            processed_data = (training, validation)

            return processed_data

        # add to the data the past lags
        data = self._data_shift.transform(data)
        past_labels = self._data_shift.past_labels

        x = None
        y = None
        x_labels = past_labels  # incluir index_labels em alguns modelos

        # adapt the data to the chosen model
        if model == 'lightgbm':
            x = data[x_labels]
            y = data[self.target_label]

            processed_data = (x, y)

        return processed_data

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

        # TFT

        # configure network and trainer
        # create dataloaders for model
        batch_size = 128
        train_dataloader = self.training.to_dataloader(train=True, batch_size=batch_size, num_workers=self.jobs)
        val_dataloader = self.validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=self.jobs)

        pl.seed_everything(42)

        tft = TemporalFusionTransformer.from_dataset(
            self.training,
            # not meaningful for finding the learning rate but otherwise very important
            learning_rate=0.03,
            hidden_size=16,  # most important hyperparameter apart from learning rate
            # number of attention heads. Set to up to 4 for large datasets
            attention_head_size=1,
            dropout=0.1,  # between 0.1 and 0.3 are good values
            hidden_continuous_size=8,  # set to <= hidden_size
            output_size=len(self.quantiles),  # 3 quantiles by default
            loss=QuantileLoss(self.quantiles),
            # reduce learning rate if no improvement in validation loss after x epochs
            reduce_on_plateau_patience=4,
        )

        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()

        trainer = pl.Trainer(
            max_epochs=25,
            gpus=0,
            weights_summary="top",
            gradient_clip_val=0.1,
            # limit_train_batches=30,  # coment in for training, running validation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=[lr_logger, early_stop_callback],
        )

        tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.3,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=len(self.quantiles),  # 3 quantiles by default
            loss=QuantileLoss(self.quantiles),
            reduce_on_plateau_patience=4,
        )

        # fit network
        trainer.fit(
            tft,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # evaluate the TFT
        
        self.evaluation_results['TFT'] = {}

        # evaluate the models on the last max lag period
        y_val = self.y[-self.oldest_lag:]
        # default values
        y_pred = tft.predict(self.validation, mode='prediction').numpy()[0]
        self.evaluation_results['TFT']['default'] = self._evaluate_model(y_val, y_pred)

        # quantile values
        y_pred = tft.predict(self.validation, mode='quantiles').numpy()[0]

        for i in range(len(self.quantiles)):
            quantile = self.quantiles[i]
            q_pred = y_pred[:, i]
            self.evaluation_results['TFT'][str(quantile)] = self._evaluate_model(y_val, q_pred, quantile)
        

        # LightGBM

        self.evaluation_results['LightGBM'] = {}

        # using quantile prediction as default
        quantile_params = {
            'objective': 'quantile',
            'metric': 'quantile',
        }
        quantile_models = [lgb.LGBMRegressor(alpha=quantil, **quantile_params)
                           for quantil in self.quantiles]

        lgbm_model = lgb.LGBMRegressor()  # Temp

        self.model = lgbm_model
        self.quantile_models = quantile_models
        self._trainer()

        # Choose the best model comparing the default prediction metric results
        min_metric = min([(x[0], x[1]['default']['wape']) for x in self.evaluation_results.items()])

        if min_metric[0] == 'LightGBM':
            self.model = lgbm_model
        elif min_metric[0] == 'TFT':
            self.model = tft

    def _trainer(self):
        """
        Train the chosen model and evaluate the final result.

        """

        if isinstance(self.model, LGBMRegressor):
            # train data shifted by the max lag period
            X_train, y_train = self.X[:-self.oldest_lag], self.y[:-self.oldest_lag]

            self.model.fit(X_train, y_train)

            for quantile_model in self.quantile_models:
                quantile_model.fit(X_train, y_train)

            # evaluate the models on the last max lag period
            X_val, y_val = self.X[-self.oldest_lag:], self.y[-self.oldest_lag:]

            # default model
            y_pred = self.model.predict(X_val)
            self.evaluation_results['LightGBM']['default'] = self._evaluate_model(y_val, y_pred)

            # quantile models
            for quantile, model in zip(self.quantiles, self.quantile_models):
                y_pred = model.predict(X_val)
                self.evaluation_results['LightGBM'][str(quantile)] = self._evaluate_model(y_val, y_pred, quantile)

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
        if isinstance(self.model, TemporalFusionTransformer):
            time_idx = list(range(len(X))) # refact to use real time idx
            X[self.index_label] = pd.to_datetime(X[self.index_label])
            X[self.index_label] = X[self.index_label].dt.tz_localize(None)
            X["time_idx"] = time_idx
            X['group_id'] = 'series'

            temp_data = self.data.iloc[-(self.oldest_lag+1):].copy()
            
            cur_X = temp_data.append(X, ignore_index=True)
            time_idx = list(range(len(cur_X))) # refact to use real time idx
            cur_X["time_idx"] = time_idx

            cur_X.index = list(range(len(cur_X)))
            
        else:
            cur_X = self._data_shift.transform(X.copy())
            cur_X = cur_X[self._data_shift.past_labels].values
        y = []

        # Prediction
        if isinstance(self.model, TemporalFusionTransformer):
            mode = 'quantiles' if quantile else 'prediction'

            date_step = cur_X[self.index_label].iloc[-1] - cur_X[self.index_label].iloc[-2]
            y = []
            for _ in range(future_steps):
                predict = self.model.predict(cur_X, mode=mode)[0][0]
                if quantile:
                    y.append(predict.numpy().tolist())
                    new_value = y[-1][1] # get quantil 0.5
                else:
                    y.append(float(predict.numpy()))
                    new_value = y[-1]

                new_date = cur_X[self.index_label].iloc[-1] + date_step                

                # auto feed the model to perform the next prediction
                new_entry = {
                    self.index_label: new_date,
                    self.target_label: new_value,
                    'time_idx': cur_X['time_idx'].iloc[-1] + 1,
                    'group_id': 'series'
                    }
                cur_X = cur_X.append(new_entry, ignore_index=True)

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

        # new_data = self.input_transformation(new_data)

        if append:
            self.data = self.data.append(new_data, ignore_index=True)
        else:
            self.data = new_data

        self.X, self.y = self._transform_data('lightgbm')

        # self._evaluate()
        self._trainer()
