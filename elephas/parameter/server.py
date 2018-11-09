import abc
import socket
from threading import Lock, Thread
import six.moves.cPickle as pickle
from flask import Flask, request
from multiprocessing import Process

from ..utils.sockets import determine_master
from ..utils.sockets import receive, send
from ..utils.serialization import dict_to_model
# from multiprocessing import Lock
from ..utils.rwlock import RWLock as Lock
from ..utils.notebook_utils import is_running_in_notebook

class BaseParameterServer(object):
    """BaseParameterServer

    Parameter servers can be started and stopped. Server implementations have
    to cater to the needs of their respective BaseParameterClient instances.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def start(self):
        """Start the parameter server instance.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self):
        """Terminate the parameter server instance.
        """
        raise NotImplementedError


class HttpServer(BaseParameterServer):
    """HttpServer

    Flask HTTP server. Defines two routes, `/parameters` to GET current
    parameters held by this server, and `/update` which can be used to
    POST updates.
    """

    def __init__(self, model, optimizer, mode, port=4000, debug=True,
                 threaded=True, use_reloader=True):
        """Initializes and HTTP server from a serialized Keras model, elephas optimizer,
        a parallelisation mode and a port to run the Flask application on. In
        hogwild mode no read- or write-locks will be acquired, in asynchronous
        mode this is the case.

        :param model: Serialized Keras model
        :param optimizer: Elephas optimizer
        :param mode: parallelization mode, either `asynchronous` or `hogwild`
        :param port: int, port to run the application on
        :param debug: boolean, Flask debug mode
        :param threaded: boolean, Flask threaded application mode
        :param use_reloader: boolean, Flask `use_reloader` argument
        """

        self.master_network = dict_to_model(model)
        self.mode = mode
        self.master_url = None
        self.optimizer = optimizer

        self.port = port

        if is_running_in_notebook():
            self.threaded = False
            self.use_reloader = False
            self.debug = False
        else:
            self.debug = debug
            self.threaded = threaded
            self.use_reloader = use_reloader

        self.lock = Lock()
        self.pickled_weights = None
        self.weights = self.master_network.get_weights()

        self.server = Process(target=self.start_flask_service)

    def start(self):
        self.server.start()
        self.master_url = determine_master(self.port)

    def stop(self):
        self.server.terminate()
        self.server.join()

    def start_flask_service(self):
        """Define Flask parameter server service.

        This HTTP server can do two things: get the current model
        parameters and update model parameters. After registering
        the `parameters` and `update` routes, the service will
        get started.

        """
        app = Flask(__name__)
        self.app = app

        @app.route('/')
        def home():
            return 'Elephas'

        @app.route('/parameters', methods=['GET'])
        def handle_get_parameters():
            if self.mode == 'asynchronous':
                self.lock.acquire_read()
            self.pickled_weights = pickle.dumps(self.weights, -1)
            pickled_weights = self.pickled_weights
            if self.mode == 'asynchronous':
                self.lock.release()
            return pickled_weights

        @app.route('/update', methods=['POST'])
        def handle_update_parameters():
            delta = pickle.loads(request.data)
            if self.mode == 'asynchronous':
                self.lock.acquire_write()

            if not self.master_network.built:
                self.master_network.build()

            base_constraint = lambda a: a
            constraints = [base_constraint for _ in self.weights]
            self.weights = self.optimizer.get_updates(self.weights, constraints, delta)
            if self.mode == 'asynchronous':
                self.lock.release()
            return 'Update done'

        self.app.run(host='0.0.0.0', debug=self.debug, port=self.port,
                     threaded=self.threaded, use_reloader=self.use_reloader)


class SocketServer(BaseParameterServer):
    """SocketServer

    A basic Python socket server

    """

    def __init__(self, model, port=4000):
        """Initializes a Socket server instance from a serializer Keras model
        and a port to listen to.

        :param model: Serialized Keras model
        :param port: int, port to run the socket on
        """

        self.model = dict_to_model(model)
        self.port = port
        self.socket = None
        self.runs = False
        self.connections = []
        self.lock = Lock()
        self.thread = None

    def start(self):
        if self.thread is not None:
            self.stop()
        self.thread = Thread(target=self.start_server)
        self.thread.start()

    def stop(self):
        self.stop_server()
        self.thread.join()
        self.thread = None

    def start_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.bind(('0.0.0.0', self.port))
        sock.listen(5)
        self.socket = sock
        self.runs = True
        self.run()

    def stop_server(self):
        self.runs = False
        if self.socket:
            for thread in self.connections:
                thread.join()
                del thread
            self.socket.close()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.connect(("localhost", self.port))
                sock.close()
            except Exception:
                pass
        self.socket = None
        self.connections = []

    def update_parameters(self, conn):
        data = receive(conn)
        delta = data['delta']
        with self.lock:
            weights = self.model.get_weights() + delta
            self.model.set_weights(weights)

    def get_parameters(self, conn):
        with self.lock:
            weights = self.model.get_weights()
        send(conn, weights)

    def action_listener(self, conn):
        while self.runs:
            get_or_update = conn.recv(1).decode()
            if get_or_update == 'u':
                self.update_parameters(conn)
            elif get_or_update == 'g':
                self.get_parameters(conn)
            else:
                raise ValueError('Received invalid action')

    def run(self):
        while self.runs:
            try:
                conn, addr = self.socket.accept()
                thread = Thread(target=self.action_listener, args=(conn, addr))
                thread.start()
                self.connections.append(thread)
            except Exception:
                print("Failed to set up socket connection.")
