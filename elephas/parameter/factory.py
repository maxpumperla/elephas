from abc import ABC, abstractmethod

from elephas.parameter import HttpClient, HttpServer, SocketClient, SocketServer


class ClientServerFactory(ABC):
    _type = 'base'

    @classmethod
    def get_factory(cls, _type):
        try:
            return next(cl for cl in cls.__subclasses__() if cl._type == _type)()
        except StopIteration:
            raise ValueError("Unknown factory type {}".format(_type))

    @abstractmethod
    def create_client(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_server(self, *args, **kwargs):
        pass


class HttpFactory(ClientServerFactory):
    _type = 'http'

    def create_client(self, *args, **kwargs):
        return HttpClient(*args, **kwargs)

    def create_server(self, *args, **kwargs):
        return HttpServer(*args, **kwargs)


class SocketFactory(ClientServerFactory):
    _type = 'socket'

    def create_client(self, *args, **kwargs):
        return SocketClient(*args, **kwargs)

    def create_server(self, *args, **kwargs):
        return SocketServer(*args, **kwargs)