import pickle

import tensorflow
from hyperopt import Trials, rand
from hyperas.ensemble import VotingModel
from hyperas.optim import get_hyperopt_model_string, base_minimizer
import numpy as np
from pyspark import SparkContext
from tensorflow.keras.models import model_from_yaml


# depend on hyperas, boto etc. is optional


class HyperParamModel(object):
    """HyperParamModel

    Computes distributed hyper-parameter optimization using Hyperas and
    Spark.
    """

    def __init__(self, sc: SparkContext, num_workers: int = 4):
        self.spark_context = sc
        self.num_workers = num_workers

    def compute_trials(self, model: tensorflow.keras.models.Model, data: np.array, max_evals: int, notebook_name=None):
        model_string = get_hyperopt_model_string(model=model, data=data, functions=None, notebook_name=notebook_name,
                                                 verbose=False, stack=3)
        hyperas_worker = HyperasWorker(model_string, max_evals)
        dummy_rdd = self.spark_context.parallelize([i for i in range(1, 1000)])
        dummy_rdd = dummy_rdd.repartition(self.num_workers)
        trials_list = dummy_rdd.mapPartitions(
            hyperas_worker._minimize).collect()

        return trials_list

    def minimize(self, model: tensorflow.keras.models.Model, data: np.array, max_evals: int, notebook_name: str = None):
        global best_model_yaml, best_model_weights

        trials_list = self.compute_trials(
            model, data, max_evals, notebook_name)

        best_val = 1e7
        for trials in trials_list:
            for trial in trials:
                val = trial.get('result').get('loss')
                if val < best_val:
                    best_val = val
                    best_model_yaml = trial.get('result').get('model')
                    best_model_weights = trial.get('result').get('weights')

        best_model = model_from_yaml(best_model_yaml)
        best_model.set_weights(pickle.loads(best_model_weights))

        return best_model

    def best_ensemble(self, nb_ensemble_models: int, model: tensorflow.keras.models.Model,
                      data: np.array, max_evals: int, voting: str = 'hard', weights=None):
        model_list = self.best_models(nb_models=nb_ensemble_models, model=model,
                                      data=data, max_evals=max_evals)
        return VotingModel(model_list, voting, weights)

    def best_models(self, nb_models, model, data, max_evals):
        trials_list = self.compute_trials(model, data, max_evals)
        num_trials = sum(len(trials) for trials in trials_list)
        if num_trials < nb_models:
            nb_models = len(trials_list)
        scores = []
        for trials in trials_list:
            scores = scores + [trial.get('result').get('loss')
                               for trial in trials]
        cut_off = sorted(scores, reverse=True)[nb_models - 1]
        model_list = []
        for trials in trials_list:
            for trial in trials:
                if trial.get('result').get('loss') >= cut_off:
                    model = model_from_yaml(trial.get('result').get('model'))
                    model.set_weights(pickle.loads(
                        trial.get('result').get('weights')))
                    model_list.append(model)
        return model_list


class HyperasWorker(object):
    """ HyperasWorker

    Executes hyper-parameter search on each worker and returns results.
    """

    def __init__(self, bc_model: tensorflow.keras.models.Model, bc_max_evals: int):
        self.model_string = bc_model
        self.max_evals = bc_max_evals

    def _minimize(self, dummy_iterator):
        trials = Trials()
        algo = rand.suggest

        elem = next(dummy_iterator)
        import random
        random.seed(elem)
        rand_seed = np.random.randint(elem)

        base_minimizer(model=None, data=None, functions=None, algo=algo, max_evals=self.max_evals,
                       trials=trials, rseed=rand_seed, full_model_string=self.model_string, notebook_name=None,
                       verbose=True, stack=3)
        yield trials
