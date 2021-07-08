import copy
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import pandas as pd
import numpy as np
from .TFTWrapper import TFTWrapper
from geneal.genetic_algorithms import ContinuousGenAlgSolver


class TFTGAWrapper(TFTWrapper):

    pbar = None

    @staticmethod
    def _evaluate(auto_ml, cur_wrapper):
        prefix = 'TFTGA'

        print(f'Evaluating {prefix}')

        solver = ContinuousGenAlgSolver(
            n_genes=8, 
            fitness_function=cur_wrapper.fitness_functions_continuous,
            pop_size=10,
            max_gen=3,
            mutation_rate=0.2,
            selection_rate=0.6,
            selection_strategy="roulette_wheel",
            problem_type=int, # Defines the possible values as int numbers
            variables_limits=[(1, 500), 
                              (1, 10), 
                              (1, 10),
                              (1, 10),
                              (1, 10),
                              (1, 100),
                              (1, 100),
                              (1, 10)] 
                              # Defines the limits of all variables
                              # Alternatively one can pass an array of tuples defining the limits
                              # for each variable: [(-10, 10), (0, 5), (0, 5), (-20, 20)]
        )

        cur_wrapper.pbar = tqdm(total=solver.pop_size*(solver.max_gen+1))

        solver.solve()

        wrapper_list = []
        y_val_matrix = auto_ml._create_validation_matrix(cur_wrapper.validation[1].values.T)

        auto_ml.evaluation_results[prefix + str(0)] = {}

        params = {
            'hidden_size': int(solver.best_individual_[0]),
            'lstm_layers': int(solver.best_individual_[1]),
            'dropout': solver.best_individual_[2]*0.1,
            'attention_head_size': int(solver.best_individual_[3]),
            'reduce_on_plateau_patience': int(solver.best_individual_[4]),
            'hidden_continuous_size': int(solver.best_individual_[5]),
            'learning_rate': int(solver.best_individual_[6])*0.001,
            'gradient_clip_val': int(solver.best_individual_[7])*0.1
        }

        cur_wrapper.train(max_epochs=50, **params)

        y_pred = np.array(cur_wrapper.predict(
                cur_wrapper.validation[0],
                future_steps=max(auto_ml.important_future_timesteps),
                history=cur_wrapper.last_period,
            ))[:, [-(n-1) for n in auto_ml.important_future_timesteps]]

        y_pred = y_pred[:-max(auto_ml.important_future_timesteps), :]

        auto_ml.evaluation_results[prefix +
                                str(0)]['default'] = auto_ml._evaluate_model(y_val_matrix.T.squeeze(), y_pred)


        wrapper_list.append(copy.copy(cur_wrapper))

        cur_wrapper.pbar.close()


        return prefix, wrapper_list


    def fitness_functions_continuous(self, individual):
        try:

            print(individual)

            params = {
                'hidden_size': int(individual[0]),
                'lstm_layers': int(individual[1]),
                'dropout': individual[2]*0.1,
                'attention_head_size': int(individual[3]),
                'reduce_on_plateau_patience': int(individual[4]),
                'hidden_continuous_size': int(individual[5]),
                'learning_rate': int(individual[6])*0.001,
                'gradient_clip_val': int(individual[7])*0.1
            }

            y_val_matrix = self.automl._create_validation_matrix(self.validation[1].values.T)

            self.train(max_epochs=50, **params)


            y_pred = np.array(self.predict(
                self.validation[0],
                future_steps=max(self.automl.important_future_timesteps),
                history=self.last_period,
            ))[:, [-(n-1) for n in self.automl.important_future_timesteps]]

            y_pred = y_pred[:-max(self.automl.important_future_timesteps), :]



            print(1/(self.automl._evaluate_model(y_val_matrix.T, y_pred))['rmse'])

            self.pbar.update(1)

            return 1/(self.automl._evaluate_model(y_val_matrix.T, y_pred))['rmse']

        except:
            return 0