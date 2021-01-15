import abc

import numpy as np
import socket
import six.moves.cPickle as pickle


try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

from ..utils.sockets import determine_master, send, receive


class BaseParameterClient(abc.ABC):
    """BaseParameterClient
    Parameter-server clients can do two things: retrieve the current parameters
    from the corresponding server, and send updates (`delta`) to the server.
    """
    client_type = 'base'

    @classmethod
    def get_client(cls, client_type: str, port: int = 4000):
        try:
            return next(cl for cl in cls.__subclasses__() if cl.client_type == client_type)(port)
        except StopIteration:
            raise ValueError("Parameter server mode has to be either `http` or `socket`, "
                             "got {}".format(client_type))

    @abc.abstractmethod
    def update_parameters(self, delta: list):
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

    client_type = 'http'

    def __init__(self, port: int = 4000):
        self.master_url = determine_master(port=port)
        self.headers = {'Content-Type': 'application/elephas'}

    def get_parameters(self):
        request = urllib2.Request('http://{}/parameters'.format(self.master_url),
                                  headers=self.headers)
        pickled_weights = urllib2.urlopen(request).read()
        return pickle.loads(pickled_weights)

    def update_parameters(self, delta: list):
        request = urllib2.Request('http://{}/update'.format(self.master_url),
                                  pickle.dumps(delta, -1), headers=self.headers)
        return urllib2.urlopen(request).read()


class SocketClient(BaseParameterClient):
    """SocketClient
    Uses a socket connection to communicate with an instance of `SocketServer`.
    The socket server listens to two types of events. Those with a `g` prefix
    indicate a get-request, those with a `u` indicate a parameter update.
    """
    client_type = 'socket'

    def __init__(self, port:int = 4000):
        self.port = port

    def get_parameters(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            host = determine_master(port=self.port).split(':')[0]
            sock.connect((host, self.port))
            sock.sendall(b'g')
            data = np.asarray(receive(sock))
        return data

    def update_parameters(self, delta: list):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            host = determine_master(port=self.port).split(':')[0]
            sock.connect((host, self.port))
            data = {'delta': delta}
            sock.sendall(b'u')
            send(sock, data)
