import numpy as np
from itertools import tee
from keras.models import model_from_yaml

from .utils import subtract_params
from .parameter import SocketClient


class SparkWorker(object):
    """Synchronous Spark worker. This code will be executed on workers.
    """
    def __init__(self, yaml, parameters, train_config, master_optimizer,
                 master_loss, master_metrics, custom_objects):
        self.yaml = yaml
        self.parameters = parameters
        self.train_config = train_config
        self.master_optimizer = master_optimizer
        self.master_loss = master_loss
        self.master_metrics = master_metrics
        self.custom_objects = custom_objects

    def train(self, data_iterator):
        """Train a keras model on a worker
        """
        feature_iterator, label_iterator = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in feature_iterator])
        y_train = np.asarray([y for x, y in label_iterator])

        model = model_from_yaml(self.yaml, self.custom_objects)
        model.compile(optimizer=self.master_optimizer,
                      loss=self.master_loss,
                      metrics=self.master_metrics)
        model.set_weights(self.parameters.value)
        weights_before_training = model.get_weights()
        if x_train.shape[0] > self.train_config.get('batch_size'):
            model.fit(x_train, y_train, **self.train_config)
        weights_after_training = model.get_weights()
        deltas = subtract_params(weights_before_training, weights_after_training)
        yield deltas


class AsynchronousSparkWorker(object):
    """Asynchronous Spark worker. This code will be executed on workers.
    """
    def __init__(self, yaml, ps_connector, train_config, frequency,
                 master_optimizer, master_loss, master_metrics,
                 custom_objects):
        self.yaml = yaml
        self.train_config = train_config
        self.frequency = frequency
        self.master_optimizer = master_optimizer
        self.master_loss = master_loss
        self.master_metrics = master_metrics
        self.custom_objects = custom_objects

    def train(self, data_iterator):
        """Train a keras model on a worker and send asynchronous updates
        to parameter server
        """
        feature_iterator, label_iterator = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in feature_iterator])
        y_train = np.asarray([y for x, y in label_iterator])

        if x_train.size == 0:
            return

        model = model_from_yaml(self.yaml, self.custom_objects)
        model.compile(optimizer=self.master_optimizer, loss=self.master_loss, metrics=self.master_metrics)

        nb_epoch = self.train_config['nb_epoch']
        batch_size = self.train_config.get('batch_size')
        nb_train_sample = x_train.shape[0]
        nb_batch = int(np.ceil(nb_train_sample / float(batch_size)))
        index_array = np.arange(nb_train_sample)
        batches = [
            (i * batch_size, min(nb_train_sample, (i + 1) * batch_size))
            for i in range(0, nb_batch)
        ]
        self.connector = SocketClient()

        if self.frequency == 'epoch':
            for epoch in range(nb_epoch):
                weights_before_training = self.connector.get_parameters()
                model.set_weights(weights_before_training)
                self.train_config['nb_epoch'] = 1
                if x_train.shape[0] > batch_size:
                    model.fit(x_train, y_train, **self.train_config)
                weights_after_training = model.get_weights()
                deltas = subtract_params(weights_before_training, weights_after_training)
                self.connector.update_parameters(deltas)
        elif self.frequency == 'batch':
            from keras.engine.training import slice_X
            for epoch in range(nb_epoch):
                if x_train.shape[0] > batch_size:
                    for (batch_start, batch_end) in batches:
                        weights_before_training = self.connector.get_parameters()
                        model.set_weights(weights_before_training)
                        batch_ids = index_array[batch_start:batch_end]
                        X = slice_X(x_train, batch_ids)
                        y = slice_X(y_train, batch_ids)
                        model.train_on_batch(X, y)
                        weights_after_training = model.get_weights()
                        deltas = subtract_params(weights_before_training, weights_after_training)
                        self.connector.update_parameters(deltas)
        else:
            raise ValueError('frequency parameter can be `epoch` or `batch, got {}'.format(self.frequency))
        yield []
