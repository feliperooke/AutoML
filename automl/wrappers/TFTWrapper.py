import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import pandas as pd
from .BaseWrapper import BaseWrapper


class TFTWrapper(BaseWrapper):
    def __init__(self, quantiles):
        super.__init__(quantiles)
        self.training = None
        self.validation = None
        self.model = None
        self.trainer = None
        self.oldest_lag = None

    def transform_data(self, data, past_lags, index_label, target_label):

        self.past_lags = past_lags
        self.oldest_lag = int(max(self.past_lags)) + 1
        self.index_label = index_label
        self.target_label = target_label

        # time index are epoch values
        # data["time_idx"] = (data[self.index_label] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        data["time_idx"] = data.index
        data['group_id'] = 'series'

        max_prediction_length = self.oldest_lag
        max_encoder_length = self.oldest_lag
        training_cutoff = data["time_idx"].max() - max_prediction_length

        self.training = TimeSeriesDataSet(
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
            lags={self.target_label: list(self.past_lags[1:-1] + 1)},
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            # allow_missings=True
        )

        # create validation set (predict=True) which means to predict the last max_prediction_length points in time
        # for each series
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, data, predict=True, stop_randomization=True)

    def train(self,
              max_epochs=25,
              hidden_size=16,
              lstm_layers=1,
              dropout=0.1,
              attention_head_size=4,
              reduce_on_plateau_patience=4,
              hidden_continuous_size=8,
              learning_rate=1e-3,
              gradient_clip_val=0.1,
              ):
        # configure network and trainer
        # create dataloaders for model
        batch_size = 128
        train_dataloader = self.training.to_dataloader(
            train=True, batch_size=batch_size)
        val_dataloader = self.validation.to_dataloader(
            train=False, batch_size=batch_size * 10)

        pl.seed_everything(42)

        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
        # lr_logger = LearningRateMonitor()

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=0,
            weights_summary=None,
            gradient_clip_val=gradient_clip_val,
            # limit_train_batches=30,  # coment in for training, running validation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=[early_stop_callback],
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            lstm_layers=lstm_layers,
            output_size=len(self.quantiles),  # 3 quantiles by default
            loss=QuantileLoss(self.quantiles),
            reduce_on_plateau_patience=reduce_on_plateau_patience,
        )

        # res = trainer.tuner.lr_find(
        #     self.model,
        #     train_dataloader=train_dataloader,
        #     val_dataloaders=val_dataloader,
        #     max_lr=10.0,
        #     min_lr=1e-6,
        # )

        # self.model = TemporalFusionTransformer.from_dataset(
        #     self.training,
        #     learning_rate=res.suggestion(), # using the suggested learining rate
        #     hidden_size=hidden_size,
        #     attention_head_size=attention_head_size,
        #     dropout=dropout,
        #     hidden_continuous_size=hidden_continuous_size,
        #     output_size=len(self.quantiles),  # 3 quantiles by default
        #     loss=QuantileLoss(self.quantiles),
        #     reduce_on_plateau_patience=reduce_on_plateau_patience,
        # )

        # fit network
        trainer.fit(
            self.model,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def predict(self, X, previous_data, future_steps, quantile=False):

        # pre-process the data
        time_idx = list(range(len(X)))  # refact to use real time idx
        X[self.index_label] = pd.to_datetime(X[self.index_label])
        X[self.index_label] = X[self.index_label].dt.tz_localize(None)
        X["time_idx"] = time_idx
        X['group_id'] = 'series'

        temp_data = previous_data.iloc[-(self.oldest_lag+1):].copy()

        cur_X = temp_data.append(X, ignore_index=True)
        time_idx = list(range(len(cur_X)))  # refact to use real time idx
        cur_X["time_idx"] = time_idx

        cur_X.index = list(range(len(cur_X)))

        mode = 'quantiles' if quantile else 'prediction'

        # interval between dates (last two dates in the dataset)
        date_step = cur_X[self.index_label].iloc[-1] - \
            cur_X[self.index_label].iloc[-2]

        y = []

        for _ in range(future_steps):
            predict = self.model.predict(cur_X, mode=mode)[0][0]
            if quantile:
                y.append(predict.numpy().tolist())
                new_value = y[-1][1]  # get quantil 0.5
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

        return y
