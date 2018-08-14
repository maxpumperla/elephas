from six.moves import cPickle as pickle
from socket import gethostbyname, gethostname


def determine_master(port=4000):
    """Determine address of master so that workers
    can connect to it.

    :param port: port on which the application runs
    :return: Master address
    """
    return gethostbyname(gethostname()) + ":" + str(port)


def _receive_all(socket, num_bytes):
    """Reads `num_bytes` bytes from the specified socket.

    :param socket: open socket instance
    :param num_bytes: number of bytes to read

    :return: received data
    """

    buffer = ''
    buffer_size = 0
    bytes_left = num_bytes
    while buffer_size < num_bytes:
        data = socket.recv(bytes_left)
        delta = len(data)
        buffer_size += delta
        bytes_left -= delta
        buffer += data
    return buffer


def receive(socket, num_bytes=20):
    """Receive data frame from open socket.

    :param socket: open socket instance
    :param num_bytes: number of bytes to read

    :return: received data
    """
    length = int(_receive_all(socket, num_bytes).decode())
    serialized_data = _receive_all(socket, length)
    return pickle.loads(serialized_data)


def send(socket, data, num_bytes=20):
    """Send data to specified socket.


    :param socket: open socket instance
    :param data: data to send
    :param num_bytes: number of bytes to read

    :return: received data
    """
    pickled_data = pickle.dumps(data, -1)
    length = str(len(pickled_data)).zfill(num_bytes)
    socket.sendall(length.encode())
    socket.sendall(pickled_data)
