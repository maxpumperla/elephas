from __future__ import print_function
from hyperopt import Trials, rand
from hyperas.ensemble import VotingModel
from hyperas.optim import get_hyperopt_model_string, base_minimizer
import numpy as np
from keras.models import model_from_yaml
import six.moves.cPickle as pickle

# depend on hyperas, boto etc. is optional

class HyperParamModel(object):
    '''
    HyperParamModel
    '''
    def __init__(self, sc, num_workers=4):
        self.spark_context = sc
        self.num_workers = num_workers

    def compute_trials(self, model, data, max_evals):
        model_string = get_hyperopt_model_string(model, data)
        bc_model = self.spark_context.broadcast(model_string)
        bc_max_evals = self.spark_context.broadcast(max_evals)

        hyperas_worker = HyperasWorker(bc_model, bc_max_evals)
        dummy_rdd = self.spark_context.parallelize([i for i in range(1, 1000)])
        dummy_rdd = dummy_rdd.repartition(self.num_workers)
        trials_list = dummy_rdd.mapPartitions(hyperas_worker.minimize).collect()

        return trials_list

    def minimize(self, model, data, max_evals):
        trials_list = self.compute_trials(model, data, max_evals)

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

    def best_ensemble(self, nb_ensemble_models, model, data, max_evals, voting='hard', weights=None):
        model_list = self.best_models(nb_models=nb_ensemble_models, model=model,
                                      data=data, max_evals=max_evals)
        return VotingModel(model_list, voting, weights)

    def best_models(self, nb_models, model, data, max_evals):
        trials_list = self.compute_trials(model, data, max_evals)
        num_trials = sum(len(trials) for trials in trials_list)
        if num_trials < nb_models:
            nb_models = len(trials)
        scores = []
        for trials in trials_list:
            scores = scores + [trial.get('result').get('loss') for trial in trials]
        cut_off = sorted(scores, reverse=True)[nb_models-1]
        model_list = []
        for trials in trials_list:
            for trial in trials:
                if trial.get('result').get('loss') >= cut_off:
                    model = model_from_yaml(trial.get('result').get('model'))
                    model.set_weights(pickle.loads(trial.get('result').get('weights')))
                    model_list.append(model)
        return model_list

class HyperasWorker(object):
    def __init__(self, bc_model, bc_max_evals):
        self.model_string = bc_model.value
        self.max_evals = bc_max_evals.value

    def minimize(self, dummy_iterator):
        trials = Trials()
        algo = rand.suggest

        elem = dummy_iterator.next()
        import random
        random.seed(elem)
        rand_seed = np.random.randint(elem)

        best_run = base_minimizer(model=None, data=None, algo=algo, max_evals=self.max_evals,
                                  trials=trials, full_model_string=self.model_string, rseed=rand_seed)
        yield trials
