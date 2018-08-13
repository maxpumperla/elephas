from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import socket
import six.moves.cPickle as pickle
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2

from ..utils.sockets import determine_master, send, receive


class BaseParameterClient(object):
    def __init__(self):
        raise NotImplementedError

    def update_parameters(self, delta):
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError


class HttpClient(BaseParameterClient):

    def __init__(self):
        self.master_url = determine_master()
        self.headers = {'Content-Type': 'application/elephas'}

    def get_parameters(self):
        '''Retrieve master weights from parameter server
        '''
        request = urllib2.Request('http://{0}/parameters'.format(self.master_url),
                                  headers=self.headers)
        pickled_weights = urllib2.urlopen(request).read()
        return pickle.loads(pickled_weights)

    def update_parameters(self, delta):
        '''Update master parameters with deltas from training process
        '''
        request = urllib2.Request('http://{0}/update'.format(self.master_url),
                                  pickle.dumps(delta, -1), headers=self.headers)
        return urllib2.urlopen(request).read()


class SocketClient(BaseParameterClient):

    def __init__(self, host='0.0.0.0', port=4000):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))

    def get_parameters(self):
        self.socket.sendall(b'g')
        print('>>> Retrieving weights from socket')
        return np.asarray(receive(self.socket))

    def update_parameters(self, delta):
        data = {}
        # data['worker_id'] = self.get_worker_id()
        data['delta'] = delta
        self.socket.sendall(b'u')
        print('>>> Start sending delta to socket')
        send(self.socket, data)
        print('>>> Done')