from automl.wrappers.SarimaWrapper import SarimaWrapper
from .SarimaWrapper import SarimaWrapper
import statsmodels.api as sm
from tqdm import tqdm
import numpy as np
from numpy.linalg import LinAlgError
import itertools
import random
import copy
from sklearn.model_selection import train_test_split
from geneal.genetic_algorithms import ContinuousGenAlgSolver

class SarimaGAWrapper(SarimaWrapper):


    @staticmethod
    def _evaluate(auto_ml, cur_wrapper):

        prefix = 'SARIMAGA'

        print(f'Evaluating {prefix}')

        solver = ContinuousGenAlgSolver(
            n_genes=7, 
            fitness_function=cur_wrapper.fitness_functions_continuous,
            pop_size=10,
            max_gen=3,
            mutation_rate=0.1,
            selection_rate=0.6,
            selection_strategy="roulette_wheel",
            problem_type=int, # Defines the possible values as int numbers
            variables_limits=[(min(cur_wrapper.past_lags), max(cur_wrapper.past_lags)), 
                              (0, 5), 
                              (min(cur_wrapper.past_lags), max(cur_wrapper.past_lags)), 
                              (min(cur_wrapper.past_lags), max(cur_wrapper.past_lags)),
                              (0, 5), 
                              (min(cur_wrapper.past_lags), max(cur_wrapper.past_lags)),
                              (min(cur_wrapper.past_lags), max(cur_wrapper.past_lags))] 
                              # Defines the limits of all variables
                              # Alternatively one can pass an array of tuples defining the limits
                              # for each variable: [(-10, 10), (0, 5), (0, 5), (-20, 20)]
        )


        solver.solve()

        params = {'order': solver.best_individual_[0:3], 'seasonal_order': solver.best_individual_[3:7]}

        cur_wrapper.train(params)

        wrapper_list = []

        y_val_matrix = auto_ml._create_validation_matrix(val_y=cur_wrapper.validation[cur_wrapper.target_label].values.T)

        auto_ml.evaluation_results[prefix + str(0)] = {}

        y_pred = np.array(cur_wrapper.predict(
            cur_wrapper.validation, max(auto_ml.important_future_timesteps)))[:, [-(n-1) for n in auto_ml.important_future_timesteps]]

        y_pred = y_pred[:-max(auto_ml.important_future_timesteps), :]
        auto_ml.evaluation_results[prefix + str(0)] = auto_ml._evaluate_model(y_val_matrix.T, y_pred)

        wrapper_list.append(copy.copy(cur_wrapper))

        return prefix, wrapper_list


    def fitness_functions_continuous(self, individual):

            try:
                params = {'order': individual[0:3], 'seasonal_order': individual[3:7]}

                y_val_matrix = self.automl._create_validation_matrix(val_y=self.validation[self.target_label].values.T)

                self.train(params)

                # self.automl.evaluation_results[prefix+str(c)] = {}

                y_pred = np.array(self.predict(self.validation, max(self.automl.important_future_timesteps)))[:, [-(n-1) for n in self.automl.important_future_timesteps]]
                y_pred = y_pred[:-max(self.automl.important_future_timesteps), :]

                print(1/(self.automl._evaluate_model(y_val_matrix.T, y_pred))['rmse'])

                return 1/(self.automl._evaluate_model(y_val_matrix.T, y_pred))['rmse']
            except:
                return 0
            