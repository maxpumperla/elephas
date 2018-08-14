from __future__ import absolute_import
from __future__ import print_function

import abc
import numpy as np
import socket
import six.moves.cPickle as pickle
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

from ..utils.sockets import determine_master, send, receive


class BaseParameterClient(object):
    """BaseParameterClient

    Parameter-server clients can do two things: retrieve the current parameters
    from the corresponding server, and send updates (`delta`) to the server.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update_parameters(self, delta):
        """Update master parameters with deltas from training process
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_parameters(self):
        """Retrieve master weights from parameter server
        """
        raise NotImplementedError


class HttpClient(BaseParameterClient):
    """HttpClient

    Uses HTTP protocol for communication with its corresponding parameter server,
    namely HttpServer. The HTTP server provides two endpoints, `/parameters` to
    get parameters and `/update` to update the server's parameters.
    """
    def __init__(self, port=4000):

        self.master_url = determine_master(port=port)
        self.headers = {'Content-Type': 'application/elephas'}

    def get_parameters(self):
        request = urllib2.Request('http://{}/parameters'.format(self.master_url),
                                  headers=self.headers)
        pickled_weights = urllib2.urlopen(request).read()
        return pickle.loads(pickled_weights)

    def update_parameters(self, delta):
        request = urllib2.Request('http://{}/update'.format(self.master_url),
                                  pickle.dumps(delta, -1), headers=self.headers)
        return urllib2.urlopen(request).read()


class SocketClient(BaseParameterClient):
    """SocketClient

    Uses a socket connection to communicate with an instance of `SocketServer`.
    The socket server listens to two types of events. Those with a `g` prefix
    indicate a get-request, those with a `u` indicate a parameter update.
    """
    def __init__(self, host='0.0.0.0', port=4000):

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))

    def get_parameters(self):
        self.socket.sendall(b'g')
        return np.asarray(receive(self.socket))

    def update_parameters(self, delta):
        data = {'delta': delta}
        self.socket.sendall(b'u')
        send(self.socket, data)
