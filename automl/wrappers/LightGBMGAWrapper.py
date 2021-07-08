from .LightGBMWrapper import LightGBMWrapper
from tqdm import tqdm
import copy
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from geneal.genetic_algorithms import ContinuousGenAlgSolver


class LightGBMGAWrapper(LightGBMWrapper):

    pbar = None

    @staticmethod
    def _evaluate(auto_ml, cur_wrapper):

        prefix = 'LightGBMGA'

        print(f'Evaluating {prefix}')

        solver = ContinuousGenAlgSolver(
            n_genes=5, 
            fitness_function=cur_wrapper.fitness_functions_continuous,
            pop_size=10,
            max_gen=20,
            mutation_rate=0.2,
            selection_rate=0.6,
            selection_strategy="roulette_wheel",
            problem_type=int, # Defines the possible values as int numbers
            variables_limits=[(10, 200), 
                              (1, 15), 
                              (1, 10),
                              (500, 15000), 
                              (50, 500)] 
                              # Defines the limits of all variables
                              # Alternatively one can pass an array of tuples defining the limits
                              # for each variable: [(-10, 10), (0, 5), (0, 5), (-20, 20)]
        )

        cur_wrapper.pbar = tqdm(total=solver.pop_size*(solver.max_gen))

        solver.solve()

        wrapper_list = []
        y_val_matrix = auto_ml._create_validation_matrix(cur_wrapper.validation[1].values.T)

        auto_ml.evaluation_results[prefix + str(0)] = {}

        params = {
            'num_leaves': int(solver.best_individual_[0]),
            'max_depth': int(solver.best_individual_[1]),
            'learning_rate': solver.best_individual_[2]*0.001,
            'num_iterations': int(solver.best_individual_[3]),
            'n_estimators': int(solver.best_individual_[4])
            }

        cur_wrapper.train(params)

        y_pred = np.array(cur_wrapper.predict(cur_wrapper.validation[0], max(auto_ml.important_future_timesteps)))[:, [-(n-1) for n in auto_ml.important_future_timesteps]]

        y_pred = y_pred[:-max(auto_ml.important_future_timesteps), :]
        auto_ml.evaluation_results[prefix + str(0)] = auto_ml._evaluate_model(y_val_matrix.T, y_pred)

        wrapper_list.append(copy.copy(cur_wrapper))

        cur_wrapper.pbar.close()

        return prefix, wrapper_list




    def fitness_functions_continuous(self, individual):
        try:

            params = {
                'num_leaves': int(individual[0]),
                'max_depth': int(individual[1]),
                'learning_rate': individual[2]*0.001,
                'num_iterations': int(individual[3]),
                'n_estimators': int(individual[4]),
            }

            y_val_matrix = self.automl._create_validation_matrix(self.validation[1].values.T)

            self.train(params)

            y_pred = np.array(self.predict(self.validation[0], max(self.automl.important_future_timesteps)))[:, [-(n-1) for n in self.automl.important_future_timesteps]]
            y_pred = y_pred[:-max(self.automl.important_future_timesteps), :]

            print(1/(self.automl._evaluate_model(y_val_matrix.T, y_pred))['rmse'])

            self.pbar.update(1)

            return 1/(self.automl._evaluate_model(y_val_matrix.T, y_pred))['rmse']

        except:
            return 0
