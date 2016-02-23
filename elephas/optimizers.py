'''
This is essentially a copy of keras' optimizers.py.
We have to modify the base class 'Optimizer' here,
as the gradients will be provided by the Spark workers,
not by one of the backends (Theano or Tensorflow).
'''
from __future__ import absolute_import
from keras import backend as K
from keras.utils.generic_utils import get_from_module
import numpy as np

from six.moves import zip


def clip_norm(g, c, n):
    ''' Clip gradients '''
    if c > 0:
        g = K.switch(K.ge(n, c), g * c / n, g)
    return g


def kl_divergence(p, p_hat):
    ''' Kullbach-Leibler divergence '''
    return p_hat - p + p * K.log(p / p_hat)


class Optimizer(object):
    '''
    Optimizer for elephas models, adapted from
    respective Keras module.
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = []

    def get_state(self):
        ''' Get latest status of optimizer updates '''
        return [u[0].get_value() for u in self.updates]

    def set_state(self, value_list):
        ''' Set current status of optimizer '''
        assert len(self.updates) == len(value_list)
        for u, v in zip(self.updates, value_list):
            u[0].set_value(v)

    def get_updates(self, params, constraints, grads):
        ''' Compute updates from gradients and constraints '''
        raise NotImplementedError

    def get_gradients(self, grads, params):

        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(g ** 2) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]

        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]

        return K.shared(grads)

    def get_config(self):
        ''' Get configuration dictionary '''
        return {"name": self.__class__.__name__}


class SGD(Optimizer):
    ''' SGD, optionally with nesterov momentum '''
    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, *args, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = 0
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

    def get_updates(self, params, constraints, grads):
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]
        new_weights = []

        for p, g, c in zip(params, grads, constraints):
            m = np.zeros_like(p)  # momentum
            v = self.momentum * m - lr * g  # velocity
            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
            new_weights.append(c(new_p))

        return new_weights

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr),
                "momentum": float(self.momentum),
                "decay": float(self.decay),
                "nesterov": self.nesterov}


class RMSprop(Optimizer):
    '''
    Reference: www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    '''
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, *args, **kwargs):
        super(RMSprop, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = lr
        self.rho = rho

    def get_updates(self, params, constraints, grads):
        accumulators = [np.zeros_like(p) for p in params]
        new_weights = []

        for p, g, a, c in zip(params, grads, accumulators, constraints):
            new_a = self.rho * a + (1 - self.rho) * g ** 2
            self.updates.append((a, new_a))

            new_p = p - self.lr * g / np.sqrt(new_a + self.epsilon)
            new_weights.append(c(new_p))

        return new_weights

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr),
                "rho": float(self.rho),
                "epsilon": self.epsilon}


class Adagrad(Optimizer):
    '''
    Reference: http://www.magicbroom.info/Papers/DuchiHaSi10.pdf
    '''
    def __init__(self, lr=0.01, epsilon=1e-6, *args, **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = lr

    def get_updates(self, params, constraints, grads):
        accumulators = [np.zeros_like(p) for p in params]
        new_weights = []

        for p, g, a, c in zip(params, grads, accumulators, constraints):
            new_a = a + g ** 2
            new_p = p - self.lr * g / np.sqrt(new_a + self.epsilon)
            new_weights.append(new_p)

        return new_weights

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr),
                "epsilon": self.epsilon}


class Adadelta(Optimizer):
    '''
        Reference: http://arxiv.org/abs/1212.5701
    '''
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, *args, **kwargs):
        super(Adadelta, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = lr

    def get_updates(self, params, constraints, grads):
        accumulators = [np.zeros_like(p) for p in params]
        delta_accumulators = [np.zeros_like(p) for p in params]
        new_weights = []

        for p, g, a, d_a, c in zip(params, grads, accumulators,
                                   delta_accumulators, constraints):
            new_a = self.rho * a + (1 - self.rho) * g ** 2
            self.updates.append((a, new_a))
            # use the new accumulator and the *old* delta_accumulator
            div = np.sqrt(new_a + self.epsilon)
            update = g * np.sqrt(d_a + self.epsilon) / div
            new_p = p - self.lr * update
            self.updates.append((p, c(new_p)))  # apply constraints

            new_weights.append(new_p)
        return new_weights

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr),
                "rho": self.rho,
                "epsilon": self.epsilon}


class Adam(Optimizer):
    '''
        Reference: http://arxiv.org/abs/1412.6980v8
        Default parameters follow those provided in the original paper.
    '''
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, *args, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = 0
        self.lr = lr

    def get_updates(self, params, constraints, grads):
        new_weights = []

        t = self.iterations + 1
        lr_t = self.lr * np.sqrt(1-self.beta_2**t)/(1-self.beta_1**t)

        for p, g, c in zip(params, grads, constraints):
            m = np.zeros_like(p)  # zero init of moment
            v = np.zeros_like(p)  # zero init of velocity

            m_t = (self.beta_1 * m) + (1 - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1 - self.beta_2) * (g**2)
            p_t = p - lr_t * m_t / (np.sqrt(v_t) + self.epsilon)
            new_weights.append(c(p_t))

        return new_weights

    def get_config(self):
        return {"name": self.__class__.__name__,
                "lr": float(self.lr),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon}

# aliases
sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'optimizer',
                           instantiate=True, kwargs=kwargs)
